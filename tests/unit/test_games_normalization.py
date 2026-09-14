import polars as pl
import pytest

from nfl_predict.data.games import normalize_games


ABBR_TO_TEAM_ID = {"DAL": "1200", "PHI": "3700", "KC": "2310", "LAC": "4400"}


def _sample_raw_schedules() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC", "2025_05_DAL_PHI"],
        "season": [2025, 2025, 2025],
        "game_type": ["REG", "REG", "WC"],
        "week": [1, 1, 5],
        "gameday": ["2025-09-04", "2025-09-05", "2025-10-05"],
        "gametime": ["20:20", "20:00", None],
        "home_team": ["PHI", "LAC", "PHI"],
        "away_team": ["DAL", "KC", "DAL"],
        "home_score": [24, 27, None],
        "away_score": [20, 21, None],
        "stadium": ["Lincoln Financial Field", "SoFi Stadium", "Lincoln Financial Field"],
    })


def test_game_id_is_passed_through_verbatim():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    assert set(normalized["game_id"].to_list()) == {"2025_01_DAL_PHI", "2025_01_KC_LAC", "2025_05_DAL_PHI"}


def test_normalization_is_deterministic():
    a = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    b = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    assert a.equals(b)


def test_team_abbreviations_resolve_to_team_ids():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_01_DAL_PHI").row(0, named=True)
    assert row["home_team_id"] == "3700"
    assert row["away_team_id"] == "1200"


def test_unresolvable_abbreviation_yields_null_team_id():
    normalized = normalize_games(_sample_raw_schedules(), {})  # empty map
    row = normalized.row(0, named=True)
    assert row["home_team_id"] is None
    assert row["away_team_id"] is None


def test_completed_game_has_final_status():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_01_DAL_PHI").row(0, named=True)
    assert row["game_status"] == "final"


def test_unplayed_game_has_scheduled_status():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_05_DAL_PHI").row(0, named=True)
    assert row["game_status"] == "scheduled"


def test_playoff_game_type_maps_to_post():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_05_DAL_PHI").row(0, named=True)
    assert row["season_type"] == "POST"


def test_kickoff_time_combines_date_and_time():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_01_DAL_PHI").row(0, named=True)
    assert row["kickoff_time_naive"] == "2025-09-04T20:20:00"


def test_null_gametime_yields_null_kickoff_time():
    normalized = normalize_games(_sample_raw_schedules(), ABBR_TO_TEAM_ID)
    row = normalized.filter(pl.col("game_id") == "2025_05_DAL_PHI").row(0, named=True)
    assert row["kickoff_time_naive"] is None


def test_missing_required_columns_raises():
    incomplete = pl.DataFrame({"game_id": ["x"], "season": [2025]})
    with pytest.raises(ValueError):
        normalize_games(incomplete, ABBR_TO_TEAM_ID)
