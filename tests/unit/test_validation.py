import polars as pl

from nfl_predict.data.validation import (
    is_promotable,
    validate_critical_columns,
    validate_normalized_games,
    validate_pbp,
    validate_pbp_schedule_linkage,
    validate_raw_teams,
    validate_row_count_anomaly,
    validate_schema_change,
)


def _valid_games() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC"],
        "season": [2025, 2025],
        "season_type": ["REG", "REG"],
        "week": [1, 1],
        "game_date": ["2025-09-04", "2025-09-05"],
        "kickoff_time_naive": ["2025-09-04T20:20:00", "2025-09-05T20:00:00"],
        "home_team_id": ["3700", "4400"],
        "away_team_id": ["1200", "2310"],
        "home_team_abbr": ["PHI", "LAC"],
        "away_team_abbr": ["DAL", "KC"],
        "home_score": [24, 27],
        "away_score": [20, 21],
        "venue": ["Lincoln Financial Field", "SoFi Stadium"],
        "game_status": ["final", "final"],
    })


def test_valid_games_have_no_issues():
    assert validate_normalized_games(_valid_games()) == []


def test_duplicate_game_id_is_fatal():
    games = pl.concat([_valid_games(), _valid_games().head(1)])
    issues = validate_normalized_games(games)
    codes = {i.code for i in issues}
    assert "duplicate_game_id" in codes
    assert not is_promotable(issues)


def test_home_equals_away_is_fatal():
    games = _valid_games().with_columns(pl.col("away_team_id").alias("home_team_id"))
    issues = validate_normalized_games(games)
    assert any(i.code == "home_equals_away" and i.level == "FATAL" for i in issues)


def test_unresolved_team_abbr_is_error_not_fatal():
    games = _valid_games()
    games = games.with_columns(
        pl.when(pl.col("game_id") == "2025_01_DAL_PHI").then(None).otherwise(pl.col("home_team_id")).alias("home_team_id")
    )
    issues = validate_normalized_games(games)
    assert any(i.code == "unresolved_team_abbr" and i.level == "ERROR" for i in issues)
    assert is_promotable(issues)  # ERROR alone doesn't block promotion


def test_impossible_week_is_error():
    games = _valid_games().with_columns(pl.lit(99).alias("week"))
    issues = validate_normalized_games(games)
    assert any(i.code == "impossible_week" for i in issues)


def test_invalid_date_is_error():
    games = _valid_games()
    games = games.with_columns(
        pl.when(pl.col("game_id") == "2025_01_DAL_PHI").then(pl.lit("not-a-date")).otherwise(pl.col("game_date")).alias("game_date")
    )
    issues = validate_normalized_games(games)
    assert any(i.code == "invalid_game_date" for i in issues)


def test_unrecognized_season_type_is_warning():
    games = _valid_games().with_columns(pl.lit("XX").alias("season_type"))
    issues = validate_normalized_games(games)
    assert any(i.code == "unrecognized_season_type" and i.level == "WARNING" for i in issues)
    assert is_promotable(issues)


def test_validate_raw_teams_flags_ambiguous_abbr():
    raw = pl.DataFrame({
        "team_abbr": ["AAA", "AAA"],
        "team_id": ["1111", "2222"],
        "team_name": ["Team One", "Team Two"],
        "team_conf": ["AFC", "NFC"],
        "team_division": ["AFC West", "NFC West"],
    })
    issues = validate_raw_teams(raw)
    assert any(i.code == "ambiguous_team_abbr_mapping" and i.level == "FATAL" for i in issues)


def test_validate_pbp_flags_duplicate_play_id():
    pbp = pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_DAL_PHI"],
        "play_id": [1.0, 1.0],
    })
    issues = validate_pbp(pbp)
    assert any(i.code == "duplicate_play_id" for i in issues)


def test_pbp_schedule_linkage_flags_orphaned_and_missing():
    issues = validate_pbp_schedule_linkage(
        pbp_game_ids={"2025_01_DAL_PHI", "2025_99_ZZZ_YYY"},
        scheduled_final_game_ids={"2025_01_DAL_PHI", "2025_02_KC_LAC"},
    )
    codes_by_game = {(i.code, i.context.get("game_id")) for i in issues}
    assert ("pbp_game_not_in_schedule", "2025_99_ZZZ_YYY") in codes_by_game
    assert ("schedule_game_missing_pbp", "2025_02_KC_LAC") in codes_by_game


def test_validate_critical_columns_missing():
    df = pl.DataFrame({"a": [1]})
    issues = validate_critical_columns("dataset_x", df, frozenset({"a", "b"}))
    assert len(issues) == 1
    assert issues[0].level == "FATAL"
    assert issues[0].context["missing_columns"] == ["b"]


def test_validate_critical_columns_none_missing():
    df = pl.DataFrame({"a": [1], "b": [2]})
    assert validate_critical_columns("dataset_x", df, frozenset({"a", "b"})) == []


def test_validate_schema_change_detects_added_and_removed():
    issues = validate_schema_change("dataset_x", previous_columns={"a", "b"}, current_columns={"b", "c"})
    codes = {i.code for i in issues}
    assert codes == {"schema_columns_added", "schema_columns_removed"}


def test_validate_schema_change_no_previous_is_silent():
    assert validate_schema_change("dataset_x", previous_columns=None, current_columns={"a"}) == []


def test_row_count_anomaly_flags_large_drop():
    issues = validate_row_count_anomaly("pbp", 2020, current_count=100, other_season_counts=[1000, 1050, 980, 1010])
    assert len(issues) == 1
    assert issues[0].level == "WARNING"


def test_row_count_anomaly_silent_when_close_to_median():
    issues = validate_row_count_anomaly("pbp", 2020, current_count=1020, other_season_counts=[1000, 1050, 980, 1010])
    assert issues == []


def test_row_count_anomaly_silent_with_too_few_other_seasons():
    issues = validate_row_count_anomaly("pbp", 2020, current_count=1, other_season_counts=[1000, 1050])
    assert issues == []
