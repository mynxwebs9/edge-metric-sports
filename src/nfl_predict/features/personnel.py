"""Leakage-safe personnel/snap continuity, from snap_counts (nflverse, 2013+ per
docs/PHASE1_DATA_REPORT.md).

`snap_counts` identifies players by `pfr_player_id` + team abbreviation, a different ID
namespace than play-by-play's `passer_player_id`/`rusher_player_id` (GSIS IDs) - see
docs/DATA_SOURCES.md's nflverse known_issues. Reconciling the two would need a player-ID
crosswalk this project hasn't built. This module avoids that problem entirely: snap
continuity is computed self-referentially within snap_counts alone (comparing one game's
roster to the game before it, both identified by the SAME pfr_player_id scheme), so no
cross-dataset ID join is needed.

Continuity is deliberately defined using only the team's own PRIOR TWO completed games -
"how much did the team's snap distribution change between its last two games" - as a
stability signal heading into the next one. It never touches the game being featured.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.nflverse_loader import SOURCE_NAME
from nfl_predict.data.raw_store import list_manifests, read_raw_snapshot

MIN_SNAP_SHARE_TO_COUNT_AS_CONTRIBUTOR = 0.0  # any recorded snap counts as "played"


def load_snap_counts_for_seasons(seasons: list[int]) -> pl.DataFrame:
    frames = []
    for m in list_manifests(SOURCE_NAME, "snap_counts"):
        if m.duplicate_of_retrieval_id is not None or len(m.requested_seasons) != 1:
            continue
        if m.requested_seasons[0] not in seasons or m.row_count == 0:
            continue
        df = read_raw_snapshot(m)
        frames.append(df.select(["game_id", "team", "pfr_player_id", "offense_snaps", "defense_snaps"]))
    if not frames:
        return pl.DataFrame(schema={"game_id": pl.Utf8, "team": pl.Utf8, "pfr_player_id": pl.Utf8, "offense_snaps": pl.Float64, "defense_snaps": pl.Float64})
    return pl.concat(frames)


def add_snap_continuity(
    team_game: pl.DataFrame, snap_counts: pl.DataFrame, abbr_to_team_id: dict[str, str],
    min_observations: int = 2,
) -> pl.DataFrame:
    snaps_with_team_id = snap_counts.with_columns(
        pl.col("team").replace_strict(abbr_to_team_id, default=None).alias("team_id")
    ).filter(pl.col("team_id").is_not_null())

    by_game_team: dict[tuple[str, str], list[dict]] = {}
    for row in snaps_with_team_id.to_dicts():
        by_game_team.setdefault((row["game_id"], row["team_id"]), []).append(row)

    out_rows: list[dict] = []
    for _, group in team_game.sort(["team_id", "sort_ts"]).group_by("team_id", maintain_order=True):
        rows = group.to_dicts()
        history: list[list[dict]] = []  # each entry: that game's list of player-snap rows

        for row in rows:
            if len(history) >= min_observations:
                prior_game = history[-1]
                two_games_ago = history[-2]
                prior_offense_ids = {r["pfr_player_id"] for r in two_games_ago if (r["offense_snaps"] or 0) > MIN_SNAP_SHARE_TO_COUNT_AS_CONTRIBUTOR}
                prior_defense_ids = {r["pfr_player_id"] for r in two_games_ago if (r["defense_snaps"] or 0) > MIN_SNAP_SHARE_TO_COUNT_AS_CONTRIBUTOR}

                total_off_snaps = sum((r["offense_snaps"] or 0) for r in prior_game)
                returning_off_snaps = sum((r["offense_snaps"] or 0) for r in prior_game if r["pfr_player_id"] in prior_offense_ids)
                total_def_snaps = sum((r["defense_snaps"] or 0) for r in prior_game)
                returning_def_snaps = sum((r["defense_snaps"] or 0) for r in prior_game if r["pfr_player_id"] in prior_defense_ids)

                row["off_snap_continuity_pct"] = returning_off_snaps / total_off_snaps if total_off_snaps else None
                row["def_snap_continuity_pct"] = returning_def_snaps / total_def_snaps if total_def_snaps else None
            else:
                row["off_snap_continuity_pct"] = None
                row["def_snap_continuity_pct"] = None

            out_rows.append(row)
            history.append(by_game_team.get((row["game_id"], row["team_id"]), []))

    return pl.DataFrame(out_rows) if out_rows else team_game
