"""Baseline 1: transparent sequential Elo-style rating model.

Requirements satisfied:
- Ratings only update after a game is complete (this module only ever consumes `targets`,
  i.e. completed games) - and updates happen strictly in kickoff order, so a game can only
  affect ratings for games that come AFTER it, never before.
- Home advantage is an explicit additive rating-points constant, not implicit.
- Postseason handling: postseason games update ratings with the SAME K-factor as regular
  season (a documented simplification - a playoff win is real evidence too, and Phase 3
  isn't attempting to model stakes-adjusted K).
- Offseason carryover: at the start of each new season, every team's rating is regressed
  toward the mean (`season_regression_fraction` of the way back to `initial_rating`) - a
  standard Elo convention, fixed (not grid-searched) at 1/3.
- Team identity: uses the Phase 1 `team_id` directly (already relocation/rebrand-stable -
  see docs/PHASE1_DATA_REPORT.md), so Elo needs no expansion/relocation mapping of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

INITIAL_RATING = 1500.0
SEASON_REGRESSION_FRACTION = 1.0 / 3.0  # fixed, documented, not grid-searched


@dataclass(frozen=True)
class EloConfig:
    k_factor: float
    home_field_advantage: float
    season_regression_fraction: float = SEASON_REGRESSION_FRACTION
    initial_rating: float = INITIAL_RATING


@dataclass
class EloModel:
    config: EloConfig
    ratings: dict[str, float] = field(default_factory=dict)
    _current_season: int | None = field(default=None, repr=False)

    def _get_rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.config.initial_rating)

    def _maybe_regress_to_mean(self, season: int) -> None:
        if self._current_season is not None and season != self._current_season:
            frac = self.config.season_regression_fraction
            for team_id in list(self.ratings):
                self.ratings[team_id] = (
                    self.config.initial_rating + (1 - frac) * (self.ratings[team_id] - self.config.initial_rating)
                )
        self._current_season = season

    def predict_pregame(self, home_team_id: str, away_team_id: str, season: int) -> dict:
        """Read-only prediction for a game that has NOT happened yet (no outcome to fold
        in) - e.g. an upcoming, unplayed matchup for live/prospective use (Phase 6). Applies
        the same season-boundary regression `run_sequential` would apply if this were the
        season's first processed game (idempotent - safe to call more than once for the
        same season), but never updates either team's rating from an outcome, since there
        isn't one yet. Uses the exact same formula `run_sequential` uses internally - this
        does not change the frozen model's behavior, it only exposes the pre-game half of
        that formula for a game with no post-game outcome to fold in."""
        self._maybe_regress_to_mean(season)
        home_r = self._get_rating(home_team_id)
        away_r = self._get_rating(away_team_id)
        elo_diff = (home_r + self.config.home_field_advantage) - away_r
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return {
            "home_team_id": home_team_id, "away_team_id": away_team_id, "season": season,
            "home_elo_pre": home_r, "away_elo_pre": away_r, "elo_diff_pre": elo_diff,
            "expected_home_win_prob": expected_home,
        }

    def run_sequential(self, games_sorted: pl.DataFrame) -> pl.DataFrame:
        """`games_sorted` must already be in strict chronological order (season, week,
        kickoff, game_id) - this method does not re-sort, so the caller is responsible for
        guaranteeing update order. Returns one row per game with PRE-game ratings and the
        PRE-game expected home win probability - i.e. exactly what would have been knowable
        before that game's outcome."""
        records = []
        for row in games_sorted.iter_rows(named=True):
            self._maybe_regress_to_mean(row["season"])
            home_r = self._get_rating(row["home_team_id"])
            away_r = self._get_rating(row["away_team_id"])
            elo_diff = (home_r + self.config.home_field_advantage) - away_r
            expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

            records.append({
                "game_id": row["game_id"], "season": row["season"], "week": row["week"],
                "home_team_id": row["home_team_id"], "away_team_id": row["away_team_id"],
                "home_elo_pre": home_r, "away_elo_pre": away_r, "elo_diff_pre": elo_diff,
                "expected_home_win_prob": expected_home,
            })

            actual = 0.5 if row["is_tie"] else float(row["home_win"])
            delta = self.config.k_factor * (actual - expected_home)
            self.ratings[row["home_team_id"]] = home_r + delta
            self.ratings[row["away_team_id"]] = away_r - delta

        return pl.DataFrame(records)


def sort_games_chronologically(targets: pl.DataFrame) -> pl.DataFrame:
    return targets.with_columns(
        pl.col("kickoff_time_naive").fill_null(pl.col("game_date") + " 00:00:00").alias("_sort_ts")
    ).sort(["season", "week", "_sort_ts", "game_id"]).drop("_sort_ts")


def fit_margin_transform(elo_history: pl.DataFrame, actual_margins: np.ndarray) -> tuple[float, float]:
    """A documented LINEAR transform from elo_diff_pre to expected margin, fit ONLY on the
    development-period Elo history passed in - not a second model, just a one-feature OLS
    (slope, intercept) satisfying "expected home margin or a documented transform to
    margin" without inventing a fixed points-per-Elo-point constant."""
    x = elo_history["elo_diff_pre"].to_numpy()
    y = np.asarray(actual_margins, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def apply_margin_transform(elo_diff: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * np.asarray(elo_diff, dtype=float) + intercept


def grid_search_elo_params(
    dev_targets_sorted: pl.DataFrame,
    k_grid: list[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    home_adv_grid: list[float] = (0.0, 25.0, 45.0, 65.0, 85.0),
) -> tuple[EloConfig, list[dict]]:
    """Picks (k_factor, home_field_advantage) by DEVELOPMENT-period sequential log loss
    only - Elo's own walk-forward predict-then-update process makes this an honest
    in-sample-of-development search, never touching the validation or sealed seasons. Small,
    predetermined grid, not an aggressive search, per the Phase 3 brief."""
    from sklearn.metrics import log_loss

    not_tied = ~dev_targets_sorted["is_tie"].to_numpy()
    results = []
    for k in k_grid:
        for home_adv in home_adv_grid:
            model = EloModel(EloConfig(k_factor=k, home_field_advantage=home_adv))
            history = model.run_sequential(dev_targets_sorted)
            # Ties (0.5, neither a win nor a loss) are excluded from log-loss scoring, same
            # as every other win-probability evaluation in this project - they still update
            # ratings (via the 0.5 actual-outcome value inside run_sequential), just aren't
            # scored as a classification outcome.
            y_true = dev_targets_sorted["home_win"].to_numpy()[not_tied].astype(float)
            y_prob = np.clip(history["expected_home_win_prob"].to_numpy()[not_tied], 1e-6, 1 - 1e-6)
            ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
            results.append({"k_factor": k, "home_field_advantage": home_adv, "dev_log_loss": ll})

    best = min(results, key=lambda r: r["dev_log_loss"])
    return EloConfig(k_factor=best["k_factor"], home_field_advantage=best["home_field_advantage"]), results
