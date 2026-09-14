# Model Specification

## Purpose

Defines what the quantitative models predict, how independent and market-aware models are
kept separate, how models are versioned, and what "the model" is allowed and not allowed to
be influenced by. This is the authority for anything that touches `src/nfl_predict/models`.
No content here is implemented yet (Phase 3+) — this document specifies the contract that
implementation must satisfy when it lands.

## The independent/market-aware split

There are always (at minimum) **two** model tracks, never one blended model:

1. **Independent model.** Inputs are limited to team/player/game data that does not derive
   from sportsbook prices: play-by-play-derived stats, rosters, schedule, weather, injuries,
   rest/travel, coaching/roster continuity, etc. This model's job is to say what a team's
   true strength and expected performance are, unaware that a market exists.
2. **Market-aware model.** Same targets, but may additionally use betting-market
   information (opening/current lines, line movement, market-implied probabilities) as
   predictive features.

Both models predict against the same games and the same targets, so they can be compared to
each other and to the market. **A feature used by the independent model must never include
sportsbook-derived data — this is enforced by the feature registry (`config/features.yaml`),
where every feature is tagged with which model track(s) it is allowed in.** Any feature
tagged for the independent model that is later found to leak market information is a bug to
fix immediately, not a judgment call.

Model outputs, model artifacts, and stored predictions must always carry a `model_track`
field (`independent` or `market_aware`) — never store an ambiguous or blended prediction.

## Prediction targets

At minimum, every model produces:

- `home_win_prob` — probability the home team wins.
- `expected_home_points`, `expected_away_points`.
- `expected_margin` (home − away).
- `expected_total` (home + away).

Derived, not separately modeled:

- `fair_moneyline_prob` — from `home_win_prob` (and its complement for away).
- `fair_spread` — from `expected_margin`.
- `fair_total` — from `expected_total`.

Derivation formulas live in code (`src/nfl_predict/models`), not duplicated in prose here —
this doc records *that* they're derived quantities, not independently fit, so nobody adds a
fifth model to predict "fair spread" directly.

## Predictive uncertainty

Point estimates alone (`expected_margin`, `expected_total`) are not sufficient for a betting
system. A trustworthy cover probability at an arbitrary sportsbook spread `S`, or an
over/under probability at an arbitrary total `T`, requires a predictive *distribution* (or a
validated, calibrated interval/quantile estimate) of the scoring margin and total — not a
made-up spread around the point estimate.

**Every model version, for both the independent and market-aware tracks, must produce or be
paired with:**

- a predictive distribution (or equivalent calibrated representation, e.g. quantiles) of
  scoring margin,
- a predictive distribution (or equivalent) of total points,
- prediction intervals at standard confidence levels derived from those distributions,
- a calibrated win probability — `P(margin > 0)` under the margin distribution, which should
  agree with (or explicitly reconcile against) the separately-produced `home_win_prob`,
- a calibrated cover probability at an arbitrary spread `S` — `P(margin > -S)` for the home
  team (sign convention: `S` is the home team's spread as quoted, e.g. `S = -3.5` means home
  favored by 3.5; the home team covers when `margin > -S`, i.e. `margin > 3.5` in that
  example) — computable for any `S`, not just today's market line,
- a calibrated over probability at an arbitrary total `T` — `P(total > T)` under the total
  distribution, computable for any `T`, not just today's market line.

**The exact method is not decided here and must not be invented as a fixed constant now**
(e.g. do not hardcode "margin residuals are Normal with σ=13.5"). Candidate approaches —
parametric residual distributions fit and validated per model family, quantile regression,
conformal prediction, or empirical backtested error distributions bucketed by situation —
are evaluated during Phase 3/4 using real out-of-sample walk-forward results, and the chosen
method is recorded here once selected, alongside the evidence that justified it. A model
version's uncertainty-estimation method is part of what identifies that version (see
Versioning below) — changing it requires a version bump like any other modeling change.

**Backtesting requirement:** an uncertainty estimate that hasn't been checked for
calibration is not usable — see `docs/BACKTESTING_RULES.md`'s predictive uncertainty
validation requirements. A model is not considered to "have" calibrated cover/over-under
probabilities until that validation has actually been run and passed.

## Baseline model families (Phase 3)

Built and compared in this order, cheapest/most-interpretable first:

1. **Naive home-field baseline** — home team wins with a fixed historical home-win rate;
   margin/total from historical league averages. No team-specific information. Exists purely
   as a floor every other model must beat.
2. **Elo-style rating model** — single continuously-updated strength rating per team,
   updated game-by-game from actual results. Cheap, no feature engineering dependency,
   useful as a second floor and as a candidate *feature* for later models.
3. **Linear/logistic statistical model** — interpretable regression over the engineered
   pregame feature set (logistic for win probability, linear for margin/total).
4. **Gradient-boosted model** (XGBoost or LightGBM) — same feature set, allowed nonlinear
   interactions. Only justified if it beats (3) on the backtesting metrics in
   `docs/BACKTESTING_RULES.md`, not merely by being fancier.

A more complex model replacing a simpler one in production must show a documented
improvement on held-out, walk-forward backtests — see `docs/BACKTESTING_RULES.md`. "It's a
better algorithm" is not sufficient justification on its own.

## Versioning

Every trained model artifact gets a version identifier: `{model_track}-{family}-v{N}`
(e.g. `independent-gbm-v3`, `market_aware-elo-v1`). A version bump is required whenever:
training data changes, feature set changes, hyperparameters change, the target definition
changes, or the predictive-uncertainty estimation method changes (see "Predictive
uncertainty" above). Version identifiers are stored:

- alongside the serialized model artifact,
- alongside every prediction that artifact produced,
- in the backtest results that evaluated it.

Nothing overwrites a previous version's artifact or its historical predictions. Superseding
a model means recording a new version, not mutating the old one.

## What the model may never take as input

- Anything that would not have been knowable at the prediction timestamp (see
  `docs/BACKTESTING_RULES.md`).
- Final game outcomes or any post-game statistic, for the game being predicted.
- LLM research-agent output as a direct numeric feature. The research agent (Phase 6) may
  inform the human/decision-engine layer *after* the model has already produced its number;
  it never feeds back into the model's inputs for that same prediction. This preserves the
  boundary in `docs/RESEARCH_AGENT.md`.

## Resolved in Phase 3

- **Gradient-boosting library: LightGBM** — chosen over XGBoost/HistGradientBoosting for
  native missing-value handling, existing project dependency status, and training speed.
  See `docs/PHASE3_MODEL_REPORT.md`.
- **Elo update constants**: K-factor=30, home-field advantage=45 rating points, selected by
  a small predetermined grid search scored on development-period (2010-2022) sequential
  log-loss only. Season-boundary mean reversion fixed at 1/3, not grid-searched. See
  `docs/PHASE3_MODEL_REPORT.md` and `src/nfl_predict/models/elo.py`.
- **Baseline model family results (2023 validation)**: see `docs/PHASE3_MODEL_REPORT.md`
  for the full comparison. Headline finding: Elo outperformed the linear and LightGBM
  models on this validation season, and the full 103-feature set underperformed a leaner
  64-feature ablation tier — "complexity must earn its place" held up as a real finding,
  not just a stated principle.
- **Predictive uncertainty**: Phase 3 began this (out-of-sample residual diagnostics for
  margin/total, computed on 2021-2022 development data — residual std, empirical
  quantiles, interval-calibration checks). Not yet validated at Phase 4's walk-forward
  scale, and no cover/over probability against a sportsbook line has been computed yet —
  both remain Phase 4 work. See `docs/PHASE3_MODEL_REPORT.md`'s "Predictive uncertainty"
  section.

## Resolved in Phase 4

- **Predictive uncertainty method**: a zero-mean Normal approximation to margin/total
  residuals, with sigma taken from out-of-sample residual diagnostics
  (`nfl_predict.models.uncertainty.compute_residual_diagnostics`) computed on a
  development-internal fit/check split (fit 2010-2020, check 2021-2022, Ridge/F_full
  features) - margin residual_std=13.059063938633658, total residual_std=14.264568098186846.
  Validated at true walk-forward scale against the sealed 2024-2025 holdout: both
  generalize well (ratio 1.02 and 0.96 respectively; see `docs/PHASE4_BACKTEST_REPORT.md`
  Step 8) - not just assumed. This IS the method now used wherever a cover/over probability
  is derived (`src/nfl_predict/market/fair_line.py`, Phase 5).
- Elo re-tuned with an expanded grid: k_factor=40, home_field_advantage=45, both interior to
  the grid (resolves the Phase 3 boundary concern) - see `docs/PHASE4_BACKTEST_REPORT.md`.
- The total-points model's Phase 3 bias was investigated and found era/distribution-shift
  related, not a stable defect - no correction applied, and the bias did not persist into
  the 2024-2025 holdout (see `docs/PHASE4_BACKTEST_REPORT.md`).
- Baseline model family results generalize to genuinely unseen data: Elo remains the best
  margin and win-probability model on the 2024-2025 walk-forward holdout, same as on the
  2023 validation season.

## Resolved in Phase 5

- **Market benchmark**: on the same 570-game 2024-2025 holdout, the market beats every
  independent candidate on margin (MAE 9.69 vs. Elo's 10.10), win probability (Brier 0.206
  vs. Elo's 0.216), and total points (MAE 10.10, corr 0.31 vs. every independent total
  model's near-zero correlation) - see `docs/PHASE5_MARKET_REPORT.md`.
- **No statistically distinguishable betting edge was found** for any independent
  candidate, on either spread or moneyline markets, across either holdout season
  individually or combined - every cover rate is at or below the standard break-even, every
  ROI confidence interval straddles zero, and the incremental-information test found no
  practically meaningful improvement from combining any independent model with the market's
  own spread.
- The market-aware model track (`docs/MODEL_SPEC.md`'s "independent/market-aware split")
  remains unbuilt as a production model - Phase 5 only ran the simple, interpretable
  incremental-information regression required to test whether one is justified, per its own
  brief's "start simple" instruction. It is not yet justified.

## Open items for a future phase

- Re-run the Phase 3 feature-family ablations inside the walk-forward framework before
  treating the single-validation-season QB-feature and full-feature-set findings as
  conclusive.
- No dedicated (non-Ridge/F_full) residual_std has been established for Elo or any other
  individual candidate - `fair_line.py` uses the one available validated estimate as an
  approximation, documented as such (see `docs/PHASE5_MARKET_REPORT.md`'s limitations).
- CLV cannot be computed until real multi-snapshot live odds data exists (Phase 5 built and
  tested the math and the append-only storage, but nothing populates it yet).
