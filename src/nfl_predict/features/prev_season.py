"""Previous-season summary features.

Deliberately a SEPARATE computation from the in-season "season-to-date" features, not a
reuse of the last game's `_season` value - a season's final `_season` rolling value
excludes that last game itself (by the same "strictly prior games" rule every other
rolling feature follows), so it understates the true full-season average by one game. This
module instead aggregates a team's ENTIRE previous season (every game, no exclusion) once,
and attaches that single constant value to every game the team plays in the following
season. Per the Phase 2 brief, this is kept as its own separate field, never blended with
current-season values using any arbitrary weighting - Phase 3 decides how (or whether) to
combine them.
"""

from __future__ import annotations

import polars as pl

PREV_SEASON_METRICS = [
    ("prev_season_off_epa_pp", "off_epa_sum", "off_plays_n"),
    ("prev_season_off_success_rate", "off_success_sum", "off_plays_n"),
    ("prev_season_def_epa_pp_allowed", "def_epa_sum", "def_plays_n"),
    ("prev_season_def_success_rate_allowed", "def_success_sum", "def_plays_n"),
]


def add_previous_season_summary(team_game: pl.DataFrame) -> pl.DataFrame:
    distinct_cols = sorted({c for _, sum_col, count_col in PREV_SEASON_METRICS for c in (sum_col, count_col)})
    full_season_totals = team_game.group_by(["team_id", "season"]).agg(
        *[pl.col(c).sum().alias(f"_total_{c}") for c in distinct_cols]
    )

    prev_season_summary = full_season_totals.with_columns(
        (pl.col("season") + 1).alias("season")
    ).rename({"team_id": "_prev_team_id"})

    exprs = []
    for out_name, sum_col, count_col in PREV_SEASON_METRICS:
        exprs.append(
            (pl.col(f"_total_{sum_col}") / pl.col(f"_total_{count_col}")).alias(out_name)
        )
    prev_season_summary = prev_season_summary.with_columns(*exprs).select(
        ["_prev_team_id", "season"] + [name for name, _, _ in PREV_SEASON_METRICS]
    )

    out = team_game.join(
        prev_season_summary, left_on=["team_id", "season"], right_on=["_prev_team_id", "season"], how="left"
    )
    return out.with_columns(
        pl.col("prev_season_off_epa_pp").is_not_null().alias("has_prev_season_data")
    )
