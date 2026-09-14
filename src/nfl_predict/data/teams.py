"""Team identity normalization.

nflverse's own `load_teams()` output already carries a stable numeric franchise identifier
(`team_id`) that correctly groups every historical abbreviation a franchise has used — e.g.
the Rams' STL / LA / LAR abbreviations, the Chargers' SD / LAC, and the Raiders' OAK / LV all
map to one team_id each. This module does not invent a new team-identity scheme; it takes
nflverse's team_id as this project's stable internal team identifier (documented, sourced
from upstream, not hand-maintained) and builds the abbr -> team_id alias map from it.

Every abbreviation seen anywhere in nflverse data (schedules, pbp, rosters, etc.) resolves
through this alias map — never through the "current" abbreviation alone. See
docs/PHASE1_DATA_REPORT.md for the exact set of aliases discovered for the 2010-2025 window.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class NormalizedTeams:
    teams: pl.DataFrame  # one row per team_id: team_id, canonical_abbr, name, nick, conference, division
    abbr_aliases: pl.DataFrame  # one row per (abbr, team_id): abbr, team_id


REQUIRED_RAW_COLUMNS = {"team_abbr", "team_id", "team_name", "team_nick", "team_conf", "team_division"}


def normalize_teams(raw_teams: pl.DataFrame, current_abbrs: set[str] | None = None) -> NormalizedTeams:
    """Build the normalized teams table and the abbr->team_id alias table.

    `current_abbrs`, if given, is the set of team abbreviations actually seen in the most
    recently ingested season's schedule — used only to pick which alias's name/division
    metadata represents a team_id's "canonical" row when a franchise has relocated. It does
    not affect the alias map itself (every alias is kept regardless), and it's a
    best-effort display convenience, never a join key — team_id is the join key everywhere.
    """
    missing = REQUIRED_RAW_COLUMNS - set(raw_teams.columns)
    if missing:
        raise ValueError(f"raw teams data is missing required columns: {sorted(missing)}")

    abbr_aliases = raw_teams.select(
        pl.col("team_abbr").alias("abbr"),
        pl.col("team_id"),
    ).unique(subset=["abbr"])

    canonical_rows: list[dict] = []
    for team_id, group in raw_teams.group_by("team_id", maintain_order=True):
        team_id = team_id[0] if isinstance(team_id, tuple) else team_id
        group_sorted = group.sort("team_abbr")
        chosen = None
        if current_abbrs:
            current_matches = group_sorted.filter(pl.col("team_abbr").is_in(list(current_abbrs)))
            if current_matches.height == 1:
                chosen = current_matches.row(0, named=True)
        if chosen is None:
            chosen = group_sorted.row(0, named=True)
        canonical_rows.append(
            {
                "team_id": team_id,
                "canonical_abbr": chosen["team_abbr"],
                "name": chosen["team_name"],
                "nickname": chosen["team_nick"],
                "conference": chosen["team_conf"],
                "division": chosen["team_division"],
            }
        )

    teams = pl.DataFrame(canonical_rows).sort("team_id")
    return NormalizedTeams(teams=teams, abbr_aliases=abbr_aliases)


def build_abbr_to_team_id(normalized: NormalizedTeams) -> dict[str, str]:
    return dict(zip(normalized.abbr_aliases["abbr"].to_list(), normalized.abbr_aliases["team_id"].to_list()))
