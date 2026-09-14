# Backtesting Rules

## Purpose

Backtesting is Phase 4, but the rules governing it constrain how Phase 1–3 code must be
written, so they're specified now. Any ingestion, storage, or feature-engineering design
that would make these rules impossible to enforce later is a design bug to fix immediately.

## The core rule: no look-ahead

A prediction generated for game G, evaluated "as of" timestamp T, may only use information
that would have actually existed and been knowable at T. This is stricter than "before the
game was played" — it means before the specific data point (injury report revision, line
movement, roster move) was actually published, not just before kickoff.

Concretely, this requires:

- **Every stored fact needs two timestamps where they can differ:** the timestamp the fact
  is *about* (e.g. "this game," "this injury report") and the timestamp it was *recorded/
  published*. A season-average stat computed today about a game two years ago is fine to
  store, but a backtest run must reconstruct what was knowable at the historical prediction
  time, not use the fully-settled, all-data-included version.
- **Rolling/aggregate features must be computed with an explicit as-of cutoff**, using only
  games strictly before that cutoff. A `groupby(season).mean()` over a full season is a
  leakage bug if it's used to predict a game from week 5 of that same season.
- **Injuries, depth charts, and odds must preserve historical revisions**, not just the
  final pre-game snapshot, if backtests are meant to simulate realistic decision timing. If
  only final snapshots are available for a given source, that limitation must be recorded
  in `docs/DATA_SOURCES.md` and the backtest must account for it (e.g. by testing a "day
  before kickoff" decision point using the best available proxy, and saying so explicitly
  in the report) rather than silently assuming perfect historical timing.
- **Season boundaries matter.** A model trained on a season must not have had access to
  any information (including future seasons' final standings, awards, etc.) that postdates
  the games it's being evaluated against in a walk-forward split.
- **Raw snapshot provenance bounds what's eligible.** Per
  `docs/ARCHITECTURE.md#raw-data-provenance`, every raw file carries a `retrieved_at`
  timestamp. A backtest reconstructing "what was knowable at T" may only use raw snapshots
  whose `retrieved_at` is on or before T; a snapshot fetched later — even of historical data
  — must not be used to compute a feature for a prediction timestamped before that fetch.
  This is what makes a backtested feature value traceable back to a specific, eligible raw
  snapshot rather than "whatever the raw store currently contains."

## Walk-forward methodology

Backtests proceed strictly forward in time:

- Never randomly shuffle games across train/test splits.
- Evaluate week-by-week (or season-by-season for coarser checks): train/fit using only data
  through week `W-1` (or earlier), predict week `W`, record the result, then advance.
- Retraining cadence (e.g. retrain every week vs. every season) must be explicit and
  documented per backtest run, not implicit.
- The Elo-style model updates continuously game-by-game by construction; other model
  families' retraining cadence must be decided and documented before backtest results are
  compared to it, otherwise the comparison is apples-to-oranges.

## Metrics

For every backtest run, report (not a subset chosen after seeing results):

- **Calibration** — predicted probabilities vs. observed frequencies, e.g. via a reliability
  plot/table by probability bucket. This covers win probability and, per "Predictive
  uncertainty validation" below, the model's cover/over-under probabilities at arbitrary
  lines and its prediction intervals — not win probability alone.
- **Log loss** and **Brier score** for win-probability predictions.
- **MAE** of predicted scoring margin vs. actual margin.
- **MAE** of predicted total vs. actual total.
- **ATS (against-the-spread) performance**, only for the sub-period where historical lines
  actually exist — never backfilled or assumed.
- **Totals betting performance**, same caveat.
- **Moneyline performance.**
- **ROI**, computed from actual historical odds where available, with the staking method
  used stated explicitly (e.g. flat stake).
- **Closing-line comparison**, where closing lines are available — how the model's implied
  line compares to where the market ultimately settled.

Every metric is reported with:

- **Sample size** (number of games) — a metric on 12 games is not treated the same as one
  on 1,200.
- **Confidence interval or comparable uncertainty measure** — not a bare point estimate.

## Predictive uncertainty validation

Per `docs/MODEL_SPEC.md`'s "Predictive uncertainty" section, a model produces a predictive
distribution (or calibrated interval/quantile estimate) for margin and total, not just a
point estimate. That estimate is not considered valid until backtesting has checked, on
held-out walk-forward data:

- **Interval calibration** — an X% prediction interval for margin (and separately for total)
  should contain the actual outcome close to X% of the time, checked empirically across the
  backtest period, not assumed from the fitting method.
- **Cover-probability calibration at arbitrary spreads** — bucket predictions by the model's
  stated cover probability (not just at the actual market line that game, but across the
  range the model can produce a probability for) and check realized cover rate per bucket,
  the same reliability-curve approach used for win probability.
- **Over/under-probability calibration at arbitrary totals** — same approach, for the
  model's stated over probability vs. realized over rate.
- These checks are reported with sample size and a confidence interval, same as every other
  metric in this document — a calibration curve built on a handful of games is not evidence.

A model's cover/over-under probabilities are not used by the decision engine
(`docs/DECISION_ENGINE.md`) until this validation has actually been run for that model
version and the results are acceptable — an unvalidated uncertainty estimate is treated the
same as not having one.

## Required comparisons

A backtest report always situates the model against:

- The naive home-field baseline.
- The Elo-style baseline.
- The sportsbook line itself (can the model beat closing lines, not just predict winners?).
- The immediately previous version of the same model (is this version actually better?).

**A model is never declared successful based on win percentage alone.** Win percentage
without calibration, sample size, and ROI-vs-market context is not evidence of anything.

## What backtesting code must never do

- Use any column that was itself computed using post-hoc/full-season aggregation without an
  as-of cutoff.
- Use final injury/odds snapshots to simulate a decision that would have been made earlier
  in the week, without explicitly flagging that approximation in the report.
- Cherry-pick a time window, bet-type subset, or metric after seeing which one looks best,
  then report only that. Reports specify the evaluation window and metric set in advance
  (or report all standard metrics/windows, not a filtered favorable subset).
- Silently drop games with missing data from a metric's sample size without stating the
  drop count and reason.
