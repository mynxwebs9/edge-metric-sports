"""Leakage-safe quarterback context.

CRITICAL RULE (see docs/PHASE2_FEATURE_REPORT.md#qb-features-and-the-leakage-rule): this
module never uses game G's own starter identity to build game G's features. "Primary QB"
is identified per game from that game's OWN play-by-play (which is only safe to do for
games strictly BEFORE the one being featured), and every QB-performance feature for game G
describes the QB who was primary in the team's most recent PRIOR game, using ONLY that QB's
own performance in his own prior primary-QB games for this team. Depth-chart-based "expected
starter" identification was investigated for Phase 2 and explicitly NOT used - see the
report for why (2025's depth_charts snapshot alone was proven structurally unusable in
Phase 1, and there is no proof that any season's depth chart reflects genuinely pregame-
available information rather than a post-hoc snapshot).

A play's QB is `passer_player_id` when present (covers pass attempts and sacks), else
`rusher_player_id` when `qb_scramble == 1` (a scramble has no passer_player_id - the QB
becomes the play's rusher). Scrambles are counted as part of a QB's own dropback stats
here (they came from a passing down), but never as part of the team-level "designed rush"
stats in pbp_aggregate.py - see that module's docstring for the same distinction from the
opposite side.
"""

from __future__ import annotations

import polars as pl

QB_ATOMIC_COLUMNS = [
    "dropback_n", "pass_epa_sum", "pass_success_sum", "pass_attempt_n", "sack_n", "int_n",
    "cpoe_sum", "cpoe_n", "scramble_n", "scramble_epa_sum",
]


def build_qb_atomic(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team_id, qb_id) - a QB's own stats in that game, for that team."""
    with_qb_id = pbp.filter(pl.col("qb_dropback") == 1).with_columns(
        pl.when(pl.col("passer_player_id").is_not_null())
        .then(pl.col("passer_player_id"))
        .otherwise(pl.col("rusher_player_id"))
        .alias("qb_id")
    )

    dropback = pl.col("qb_dropback") == 1  # always true here, kept for clarity of intent
    pass_attempt = pl.col("pass_attempt") == 1
    is_scramble = pl.col("qb_scramble") == 1

    return (
        with_qb_id.filter(pl.col("qb_id").is_not_null())
        .group_by(["game_id", "posteam_id", "qb_id"])
        .agg(
            dropback.sum().alias("dropback_n"),
            pl.col("epa").sum().alias("pass_epa_sum"),
            pl.col("success").sum().alias("pass_success_sum"),
            pass_attempt.sum().alias("pass_attempt_n"),
            pl.col("sack").sum().alias("sack_n"),
            pl.col("interception").sum().alias("int_n"),
            pl.col("cpoe").filter(pl.col("cpoe").is_not_null()).sum().alias("cpoe_sum"),
            pl.col("cpoe").is_not_null().sum().alias("cpoe_n"),
            is_scramble.sum().alias("scramble_n"),
            pl.col("epa").filter(is_scramble).sum().alias("scramble_epa_sum"),
        )
        .rename({"posteam_id": "team_id"})
    )


def identify_primary_qb(qb_atomic: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team_id) -> the qb_id with the most dropbacks in that game for
    that team. Ties broken by qb_id string for determinism (extremely rare in practice)."""
    return (
        qb_atomic.sort(["game_id", "team_id", "dropback_n", "qb_id"], descending=[False, False, True, True])
        .group_by(["game_id", "team_id"], maintain_order=True)
        .first()
        .select(["game_id", "team_id", "qb_id"])
        .rename({"qb_id": "primary_qb_id"})
    )


def build_qb_features(
    team_game: pl.DataFrame,
    qb_atomic: pl.DataFrame,
    primary_qb: pl.DataFrame,
    min_observations: int = 1,
) -> pl.DataFrame:
    """Adds, per (team_id, game): the identity and continuity of the team's PREVIOUS game's
    primary QB, plus that QB's own season-to-date rolling stats from his own prior
    primary-QB starts for this team (season-to-date, reset per season - consistent with
    every other season-to-date feature in this project)."""
    qb_atomic_by_key = {
        (row["game_id"], row["team_id"], row["qb_id"]): row
        for row in qb_atomic.to_dicts()
    }
    primary_by_game_team = {
        (row["game_id"], row["team_id"]): row["primary_qb_id"]
        for row in primary_qb.to_dicts()
    }

    stat_cols = ("qb_epa_dropback_season", "qb_success_rate_season", "qb_sack_rate_season",
                 "qb_int_rate_season", "qb_cpoe_season", "qb_scramble_epa_season")

    out_rows: list[dict] = []
    for _, group in team_game.sort(["team_id", "sort_ts"]).group_by("team_id", maintain_order=True):
        rows = group.to_dicts()
        current_season: int | None = None
        # qb_id -> list of that QB's own atomic dict, only for games in the CURRENT season
        # where he was this team's primary QB, in chronological order.
        season_qb_history: dict[str, list[dict]] = {}
        prev_game_primary_qb: str | None = None
        # Tracks how many consecutive games (ending at the most recently processed game)
        # `streak_qb` has been primary - a simple forward-running counter, never a backward
        # re-scan, so it can't accidentally look past the current position.
        streak_qb: str | None = None
        streak_count = 0

        for row in rows:
            if row["season"] != current_season:
                current_season = row["season"]
                season_qb_history = {}
                prev_game_primary_qb = None
                streak_qb, streak_count = None, 0

            row["qb_primary_id"] = prev_game_primary_qb
            row["qb_consecutive_starts"] = streak_count

            his_games = season_qb_history.get(prev_game_primary_qb, []) if prev_game_primary_qb else []
            row["qb_starts_season"] = len(his_games)
            if prev_game_primary_qb is not None and len(his_games) >= min_observations:
                total_dropbacks = sum(g["dropback_n"] for g in his_games)
                total_pass_attempts = sum(g["pass_attempt_n"] for g in his_games)
                total_cpoe_n = sum(g["cpoe_n"] for g in his_games)
                row["qb_epa_dropback_season"] = sum(g["pass_epa_sum"] for g in his_games) / total_dropbacks if total_dropbacks else None
                row["qb_success_rate_season"] = sum(g["pass_success_sum"] for g in his_games) / total_dropbacks if total_dropbacks else None
                row["qb_sack_rate_season"] = sum(g["sack_n"] for g in his_games) / total_pass_attempts if total_pass_attempts else None
                row["qb_int_rate_season"] = sum(g["int_n"] for g in his_games) / total_pass_attempts if total_pass_attempts else None
                row["qb_cpoe_season"] = sum(g["cpoe_sum"] for g in his_games) / total_cpoe_n if total_cpoe_n else None
                row["qb_scramble_epa_season"] = sum(g["scramble_epa_sum"] for g in his_games) / total_dropbacks if total_dropbacks else None
            else:
                for col in stat_cols:
                    row[col] = None

            # Now advance state using THIS game's own primary QB - never used above, only
            # recorded for the NEXT row's "previous game" lookup.
            this_game_primary = primary_by_game_team.get((row["game_id"], row["team_id"]))
            if this_game_primary == streak_qb:
                streak_count += 1
            else:
                streak_qb = this_game_primary
                streak_count = 1 if this_game_primary is not None else 0
            if this_game_primary is not None:
                atomic = qb_atomic_by_key.get((row["game_id"], row["team_id"], this_game_primary))
                if atomic is not None:
                    season_qb_history.setdefault(this_game_primary, []).append(atomic)
            prev_game_primary_qb = this_game_primary

            out_rows.append(row)

    return pl.DataFrame(out_rows) if out_rows else team_game
