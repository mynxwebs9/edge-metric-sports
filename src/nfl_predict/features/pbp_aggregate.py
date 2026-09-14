"""Per-(game, team) play-by-play aggregation: offense and defense, ALL-PLAY and
COMPETITIVE-PLAY variants.

Produces one row per (game_id, team_id) with raw SUMS and COUNTS (not rates) for each stat
family - e.g. `off_epa_sum` and `off_plays_n`, not `off_epa_per_play`. Rates are computed
later, AFTER rolling sums/counts across a window of prior games (see rolling.py) - this is
deliberate: averaging a team's per-game EPA/play across games with different play counts
would silently under-weight high-snap games, whereas summing EPA and dividing by summed
plays across the window gives the statistically correct play-weighted rate.

This module only reads the normalized play_by_play Parquet for the game's OWN season/game -
it has no notion of "prior games" at all. That's the rolling module's job. This keeps the
leakage boundary crisp: this module only ever looks at a game's own plays, and the rolling
module only ever looks at OTHER games' already-computed atomic rows.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.config import get_feature_engine_config

# Plays that count toward "offensive/defensive plays" at all. Excludes kickoffs, punts,
# field goals, extra points, no-plays (penalties with no snap outcome), AND qb_kneel/
# qb_spike - kneels and spikes are clock-management plays, not representative offensive
# performance, and this play_type filter excludes them automatically (their play_type is
# "qb_kneel"/"qb_spike", never "pass"/"run") - see docs/PHASE2_FEATURE_REPORT.md's
# "Garbage time / game state" section for why this is a deliberate decision, not an
# oversight.
_OFFENSE_PLAY_TYPES = ["pass", "run"]


def _competitive_filter(threshold: float) -> pl.Expr:
    return pl.col("home_wp").is_between(threshold, 1 - threshold)


def aggregate_team_game_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Returns one row per (game_id, team_id, side) atomic stat row, side in {"off","def"}.

    Callers combine the two sides per team-game (see `combine_offense_defense`). Kept
    separate here because the aggregation logic (group by posteam vs defteam) is identical
    modulo which column is grouped on and how columns are named.
    """
    cfg = get_feature_engine_config()
    comp_filter = _competitive_filter(cfg["garbage_time_wp_threshold"])
    explosive_pass_yards = cfg["explosive_pass_yards_gained"]
    explosive_rush_yards = cfg["explosive_rush_yards_gained"]
    redzone_max = cfg["redzone_yardline_100_max"]

    base = pbp.filter(pl.col("play_type").is_in(_OFFENSE_PLAY_TYPES))

    designed_rush = pl.col("rush_attempt") == 1  # qb_scramble excluded via separate flag below
    is_scramble = pl.col("qb_scramble") == 1
    designed_rush_only = designed_rush & ~is_scramble
    dropback = pl.col("qb_dropback") == 1
    pass_attempt = pl.col("pass_attempt") == 1
    early_down = pl.col("down").is_in([1, 2])
    third_down = pl.col("down") == 3

    def _agg_for(group_col: str, prefix: str) -> pl.DataFrame:
        return (
            base.group_by(["game_id", group_col])
            .agg(
                pl.len().alias(f"{prefix}_plays_n"),
                pl.col("epa").sum().alias(f"{prefix}_epa_sum"),
                pl.col("success").sum().alias(f"{prefix}_success_sum"),
                comp_filter.sum().alias(f"{prefix}_plays_n_comp"),
                pl.col("epa").filter(comp_filter).sum().alias(f"{prefix}_epa_sum_comp"),
                pl.col("success").filter(comp_filter).sum().alias(f"{prefix}_success_sum_comp"),

                dropback.sum().alias(f"{prefix}_dropback_n"),
                pl.col("epa").filter(dropback).sum().alias(f"{prefix}_pass_epa_sum"),
                pl.col("success").filter(dropback).sum().alias(f"{prefix}_pass_success_sum"),
                pl.col("yards_gained").filter(dropback).sum().alias(f"{prefix}_dropback_yards_sum"),

                designed_rush_only.sum().alias(f"{prefix}_rush_n"),
                pl.col("epa").filter(designed_rush_only).sum().alias(f"{prefix}_rush_epa_sum"),
                pl.col("success").filter(designed_rush_only).sum().alias(f"{prefix}_rush_success_sum"),
                pl.col("yards_gained").filter(designed_rush_only).sum().alias(f"{prefix}_rush_yards_sum"),
                (designed_rush_only & (pl.col("yards_gained") >= explosive_rush_yards)).sum().alias(f"{prefix}_explosive_rush_n"),

                pass_attempt.sum().alias(f"{prefix}_pass_attempt_n"),
                (pass_attempt & (pl.col("complete_pass") == 1)).sum().alias(f"{prefix}_complete_n"),
                (pass_attempt & (pl.col("yards_gained") >= explosive_pass_yards) & (pl.col("complete_pass") == 1)).sum().alias(f"{prefix}_explosive_pass_n"),
                (pass_attempt & (pl.col("qb_hit") == 1)).sum().alias(f"{prefix}_qb_hit_n"),
                pl.col("sack").sum().alias(f"{prefix}_sack_n"),
                pl.col("interception").sum().alias(f"{prefix}_int_n"),
                pl.col("fumble_lost").sum().alias(f"{prefix}_fumble_lost_n"),
                pl.col("cpoe").filter(pl.col("cpoe").is_not_null()).sum().alias(f"{prefix}_cpoe_sum"),
                pl.col("cpoe").is_not_null().sum().alias(f"{prefix}_cpoe_n"),

                is_scramble.sum().alias(f"{prefix}_scramble_n"),
                pl.col("epa").filter(is_scramble).sum().alias(f"{prefix}_scramble_epa_sum"),

                early_down.sum().alias(f"{prefix}_early_down_n"),
                pl.col("epa").filter(early_down).sum().alias(f"{prefix}_early_down_epa_sum"),
                pl.col("success").filter(early_down).sum().alias(f"{prefix}_early_down_success_sum"),

                third_down.sum().alias(f"{prefix}_third_down_att_n"),
                (third_down & (pl.col("third_down_converted") == 1)).sum().alias(f"{prefix}_third_down_conv_n"),
            )
            .rename({group_col: "team_id"})
        )

    off = _agg_for("posteam_id", "off")
    deff = _agg_for("defteam_id", "def")

    redzone_off = _redzone_trips(pbp, "posteam_id", redzone_max).rename({"trip_team": "team_id"})
    redzone_def = _redzone_trips(pbp, "defteam_id", redzone_max).rename({
        "trip_team": "team_id", "redzone_trips_n": "redzone_trips_n_allowed", "redzone_td_n": "redzone_td_n_allowed",
    })

    off = off.join(redzone_off, on=["game_id", "team_id"], how="left")
    deff = deff.join(redzone_def, on=["game_id", "team_id"], how="left")

    return off.join(deff, on=["game_id", "team_id"], how="full", coalesce=True)


def _redzone_trips(pbp: pl.DataFrame, team_col: str, redzone_max: int) -> pl.DataFrame:
    """One row per (game_id, drive) that reached the red zone, then rolled up to a
    per-(game, team) trip count and TD count. `drive_inside20`/`fixed_drive_result` are
    nflverse-provided drive-level fields - a "red zone trip" is a drive that ever got to
    yardline_100 <= redzone_max while that team had the ball; the trip counts as a
    touchdown if the drive's fixed_drive_result is "Touchdown"."""
    drives = (
        pbp.filter(pl.col(team_col).is_not_null() & pl.col("fixed_drive").is_not_null())
        .group_by(["game_id", team_col, "fixed_drive"])
        .agg(
            (pl.col("yardline_100") <= redzone_max).any().alias("reached_redzone"),
            (pl.col("fixed_drive_result") == "Touchdown").any().alias("drive_is_td"),
        )
        .filter(pl.col("reached_redzone"))
        .group_by(["game_id", team_col])
        .agg(
            pl.len().alias("redzone_trips_n"),
            pl.col("drive_is_td").sum().alias("redzone_td_n"),
        )
        .rename({team_col: "trip_team"})
    )
    return drives
