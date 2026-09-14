"""Leakage-safe opponent-quality context.

Deliberately NOT a learned opponent-adjustment model (per the Phase 2 brief: "avoid a
complicated learned opponent-adjustment algorithm... document as deferred to Phase 3 rather
than improvise one"). This is a simple average of each opponent's OWN pregame rating at the
time T played them - and that opponent rating is itself already leakage-safe, because it's
just that opponent's `off_epa_pp_season`/`def_epa_pp_allowed_season` value for THEIR game
against T (computed by rolling.py the same way as every other team's season-to-date value -
i.e., using only games before that specific meeting). Averaging already-safe per-meeting
values across T's own prior meetings introduces no new leakage: nothing about a game after
T's current game (or about T itself) enters an opponent's rating.

Must run AFTER team_game's own off_epa_pp_season/def_epa_pp_allowed_season columns exist.
"""

from __future__ import annotations

import polars as pl


def add_opponent_quality_faced(team_game: pl.DataFrame) -> pl.DataFrame:
    opponent_rating_per_meeting = team_game.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opponent_id"),
        pl.col("off_epa_pp_season").alias("_opp_off_rating_at_meeting"),
        pl.col("def_epa_pp_allowed_season").alias("_opp_def_rating_at_meeting"),
    )

    with_opp_rating = team_game.join(
        opponent_rating_per_meeting, on=["game_id", "opponent_id"], how="left"
    )

    out_rows: list[dict] = []
    for _, group in with_opp_rating.sort(["team_id", "sort_ts"]).group_by("team_id", maintain_order=True):
        rows = group.to_dicts()
        current_season: int | None = None
        off_ratings_faced: list[float] = []
        def_ratings_faced: list[float] = []

        for row in rows:
            if row["season"] != current_season:
                current_season = row["season"]
                off_ratings_faced, def_ratings_faced = [], []

            row["opp_off_epa_faced_season"] = (
                sum(off_ratings_faced) / len(off_ratings_faced) if off_ratings_faced else None
            )
            row["opp_def_epa_faced_season"] = (
                sum(def_ratings_faced) / len(def_ratings_faced) if def_ratings_faced else None
            )

            if row["_opp_off_rating_at_meeting"] is not None:
                off_ratings_faced.append(row["_opp_off_rating_at_meeting"])
            if row["_opp_def_rating_at_meeting"] is not None:
                def_ratings_faced.append(row["_opp_def_rating_at_meeting"])

            out_rows.append(row)

    result = pl.DataFrame(out_rows) if out_rows else with_opp_rating
    return result.drop(["_opp_off_rating_at_meeting", "_opp_def_rating_at_meeting"])
