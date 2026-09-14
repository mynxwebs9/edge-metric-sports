import polars as pl
import pytest

from nfl_predict.data.teams import build_abbr_to_team_id, normalize_teams


def _sample_raw_teams() -> pl.DataFrame:
    # Mimics nflreadpy's load_teams() shape: one row per historical abbreviation, with a
    # relocated franchise (team_id "2510") appearing under three abbreviations, matching
    # the real Rams STL/LA/LAR pattern discovered against live data (see
    # docs/PHASE1_DATA_REPORT.md).
    return pl.DataFrame({
        "team_abbr": ["ARI", "STL", "LA", "LAR", "KC"],
        "team_id": ["3800", "2510", "2510", "2510", "2310"],
        "team_name": [
            "Arizona Cardinals", "St. Louis Rams", "Los Angeles Rams", "Los Angeles Rams",
            "Kansas City Chiefs",
        ],
        "team_nick": ["Cardinals", "Rams", "Rams", "Rams", "Chiefs"],
        "team_conf": ["NFC", "NFC", "NFC", "NFC", "AFC"],
        "team_division": [
            "NFC West", "NFC West", "NFC West", "NFC West", "AFC West",
        ],
    })


def test_normalize_teams_produces_one_row_per_team_id():
    normalized = normalize_teams(_sample_raw_teams())
    assert normalized.teams.height == 3  # ARI, Rams (collapsed), KC
    assert set(normalized.teams["team_id"].to_list()) == {"3800", "2510", "2310"}


def test_normalize_teams_keeps_every_alias():
    normalized = normalize_teams(_sample_raw_teams())
    assert normalized.abbr_aliases.height == 5
    assert set(normalized.abbr_aliases["abbr"].to_list()) == {"ARI", "STL", "LA", "LAR", "KC"}


def test_all_rams_aliases_resolve_to_same_team_id():
    normalized = normalize_teams(_sample_raw_teams())
    abbr_to_id = build_abbr_to_team_id(normalized)
    assert abbr_to_id["STL"] == abbr_to_id["LA"] == abbr_to_id["LAR"] == "2510"


def test_canonical_abbr_prefers_current_abbrs_hint():
    normalized = normalize_teams(_sample_raw_teams(), current_abbrs={"LAR", "ARI", "KC"})
    rams_row = normalized.teams.filter(pl.col("team_id") == "2510").row(0, named=True)
    assert rams_row["canonical_abbr"] == "LAR"


def test_canonical_abbr_falls_back_deterministically_without_hint():
    normalized_a = normalize_teams(_sample_raw_teams())
    normalized_b = normalize_teams(_sample_raw_teams())
    rams_a = normalized_a.teams.filter(pl.col("team_id") == "2510").row(0, named=True)
    rams_b = normalized_b.teams.filter(pl.col("team_id") == "2510").row(0, named=True)
    assert rams_a["canonical_abbr"] == rams_b["canonical_abbr"]  # deterministic, not random


def test_missing_required_columns_raises():
    incomplete = pl.DataFrame({"team_abbr": ["ARI"], "team_id": ["3800"]})
    with pytest.raises(ValueError):
        normalize_teams(incomplete)
