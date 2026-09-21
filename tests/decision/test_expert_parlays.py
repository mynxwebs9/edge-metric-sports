"""Expert parlays: a parlay wins only if EVERY leg wins, and the public record must never be
flattered by a guess. These tests prove the publish-time rules (same pre-kickoff rule as
single picks, players resolved from the recorded box-score data or refused, legs frozen into
the ledger) and the grading rules (loses as soon as any leg loses, wins only on all legs,
stays pending on a stale box score, and hands anything ambiguous - a push, a player with no
stat row - to a human instead of guessing)."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.player_stats import PlayerStatsUnavailableError
from nfl_predict.decision import expert_parlays, parlay_settlement
from nfl_predict.decision.expert_parlays import build_expert_parlay, publish_expert_parlay
from nfl_predict.decision.expert_picks import ExpertPickError
from nfl_predict.decision.pick_ledger import read_current_picks
from nfl_predict.decision.pick_settlement import settle_all_pending_picks
from nfl_predict.decision.records import compute_category_record

GAME_ID = "2026_02_NYG_LA"
OTHER_GAME_ID = "2026_03_LA_DEN"
SEASON = 2026
BEFORE_KICKOFF = "2026-09-20T12:00:00+00:00"  # NYG@LA kicks off 2026-09-21 20:15 ET = 2026-09-22T00:15Z
AFTER_KICKOFF = "2026-09-22T01:00:00+00:00"
STAFFORD, SKATTEBO = "00-0026498", "00-0040715"

LEGS = [
    {"type": "moneyline", "game_id": GAME_ID, "team": "LA", "price": -305},
    {"type": "player_prop", "game_id": GAME_ID, "player": "Matthew Stafford", "stat": "passing_yards", "at_least": 210, "price": -233},
    {"type": "player_prop", "game_id": GAME_ID, "player": "Cam Skattebo", "stat": "rushing_yards", "at_least": 40, "price": -242},
]

_SCHEMA = {
    "player_id": pl.Utf8, "player_display_name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8, "week": pl.Int64,
    "season_type": pl.Utf8, "passing_yards": pl.Int64, "rushing_yards": pl.Int64,
}


def _row(player_id, name, pos, team, week, passing=0, rushing=0):
    return {"player_id": player_id, "player_display_name": name, "position": pos, "team": team, "week": week,
            "season_type": "REG", "passing_yards": passing, "rushing_yards": rushing}


def stats_frame(week2: bool = False, stafford=None, skattebo=None, skattebo_row=True) -> pl.DataFrame:
    """Week 1 rows (so players can be resolved) and, optionally, the Week 2 box score. A
    Week 2 frame always includes a filler row per team so 'the box score is ingested' is
    distinguishable from 'this player has no row'."""
    rows = [_row(STAFFORD, "Matthew Stafford", "QB", "LA", 1, passing=155, rushing=-1),
            _row(SKATTEBO, "Cam Skattebo", "RB", "NYG", 1, rushing=81)]
    if week2:
        rows += [_row("00-filler-la", "Filler LA", "WR", "LA", 2), _row("00-filler-nyg", "Filler NYG", "WR", "NYG", 2)]
        rows.append(_row(STAFFORD, "Matthew Stafford", "QB", "LA", 2, passing=stafford))
        if skattebo_row:
            rows.append(_row(SKATTEBO, "Cam Skattebo", "RB", "NYG", 2, rushing=skattebo))
    return pl.DataFrame(rows, schema=_SCHEMA)


@pytest.fixture
def seeded(isolated_data_dir, monkeypatch):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [("2510", "LA", "Los Angeles Rams", "Rams"), ("3410", "NYG", "New York Giants", "Giants"), ("1400", "DEN", "Denver Broncos", "Broncos")],
    )
    for game_id, week, kickoff, home, away, home_id, away_id in (
        (GAME_ID, 2, "2026-09-21T20:15:00", "LA", "NYG", "2510", "3410"),
        (OTHER_GAME_ID, 3, "2026-09-27T20:20:00", "DEN", "LA", "1400", "2510"),
    ):
        conn.execute(
            "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
            "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
            "VALUES (?, ?, 'REG', ?, ?, ?, ?, ?, ?, ?, 'scheduled', '2026-09-16T00:00:00')",
            (game_id, SEASON, week, kickoff[:10], kickoff, home_id, away_id, home, away),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(expert_parlays, "load_player_week_stats", lambda season: stats_frame())
    return isolated_data_dir


def _finish(game_id, home_score, away_score):
    conn = get_connection()
    conn.execute("UPDATE games SET home_score = ?, away_score = ?, game_status = 'final' WHERE game_id = ?", (home_score, away_score, game_id))
    conn.commit()
    conn.close()


def _settle(monkeypatch, frame):
    monkeypatch.setattr(parlay_settlement, "load_player_week_stats", lambda season: frame)
    return settle_all_pending_picks(now="2026-09-22T05:00:00+00:00")


# ---------------------------------------------------------------- publishing

def test_builds_the_real_three_leg_parlay_with_legs_frozen_in_plain_english(seeded):
    pick = publish_expert_parlay(LEGS, 180, note="Rams roll.", now=BEFORE_KICKOFF)

    assert (pick.category, pick.market_type, pick.price, pick.note) == ("EXPERT_PARLAYS", "parlay", 180, "Rams roll.")
    assert pick.kickoff_at == "2026-09-22T00:15:00+00:00"
    assert [leg["description"] for leg in pick.legs] == [
        "LA moneyline", "Matthew Stafford 210+ passing yards", "Cam Skattebo 40+ rushing yards",
    ]
    stafford = pick.legs[1]
    assert (stafford["player_id"], stafford["team"], stafford["direction"], stafford["line"]) == (STAFFORD, "LA", "over", 209.5)
    assert pick.legs[0]["matchup"] == "NYG @ LA"
    assert read_current_picks()[0]["legs"] == pick.legs  # stored exactly as published


def test_a_spread_leg_is_stored_in_the_home_spread_convention(seeded):
    """LA +3 on the road at DEN means the HOME (DEN) spread is -3.0."""
    legs = [
        {"type": "moneyline", "game_id": GAME_ID, "team": "LA", "price": -305},
        {"type": "spread", "game_id": OTHER_GAME_ID, "team": "LA", "line": 3, "price": -110},
    ]
    pick = build_expert_parlay(legs, 250, now=BEFORE_KICKOFF)
    spread = pick.legs[1]
    assert (spread["selection"], spread["line"], spread["description"]) == ("away", -3.0, "LA +3")
    assert pick.kickoff_at == "2026-09-22T00:15:00+00:00"  # the earliest leg's kickoff


def test_a_line_and_direction_can_be_given_instead_of_at_least(seeded):
    legs = [LEGS[0], {"type": "player_prop", "game_id": GAME_ID, "player": "Matthew Stafford", "stat": "passing_yards",
                      "direction": "under", "line": 260.5}]
    assert build_expert_parlay(legs, 200, now=BEFORE_KICKOFF).legs[1]["description"] == "Matthew Stafford under 260.5 passing yards"


@pytest.mark.parametrize("mutate,match", [
    (lambda legs: legs[:1], "needs 2-12 legs"),
    (lambda legs: [legs[0], {**legs[1], "player": "Nobody Real"}], "No player named"),
    (lambda legs: [legs[0], {**legs[1], "stat": "sacks"}], "not supported"),
    (lambda legs: [legs[0], {**legs[1], "line": 209.5}], "exactly one of"),
    (lambda legs: [legs[0], {k: v for k, v in legs[1].items() if k != "at_least"}], "exactly one of"),
    (lambda legs: [legs[0], {**legs[1], "at_least": None, "line": 209.5, "direction": "around"}], "'over' or 'under'"),
    (lambda legs: [legs[0], legs[0]], "more than once"),
    (lambda legs: [legs[0], {**legs[0], "team": "NYG"}], "both sides"),
    (lambda legs: [{**legs[0], "team": "ZZZ"}, legs[1]], "not in"),
    (lambda legs: [{**legs[0], "game_id": "2026_02_NOPE_NOPE"}, legs[1]], "not in the 2026 schedule"),
    (lambda legs: [{**legs[0], "type": "total"}, legs[1]], "not supported"),
    (lambda legs: [{**legs[0], "price": -50}, legs[1]], "not valid American odds"),
])
def test_bad_parlays_are_refused_with_a_clear_reason_and_nothing_is_written(seeded, mutate, match):
    with pytest.raises(ExpertPickError, match=match):
        publish_expert_parlay(mutate(LEGS), 180, now=BEFORE_KICKOFF)
    assert read_current_picks() == []


def test_a_bad_parlay_price_is_refused(seeded):
    with pytest.raises(ExpertPickError, match="not valid American odds"):
        publish_expert_parlay(LEGS, 50, now=BEFORE_KICKOFF)


def test_a_leg_whose_game_has_kicked_off_refuses_the_whole_parlay(seeded):
    with pytest.raises(ExpertPickError, match="kicked off"):
        publish_expert_parlay(LEGS, 180, now=AFTER_KICKOFF)
    assert read_current_picks() == []


def test_a_prop_leg_needs_the_box_score_data_to_be_ingested(seeded, monkeypatch):
    def _not_ingested(season):
        raise PlayerStatsUnavailableError("No ingested player_stats snapshot for season=2026")

    monkeypatch.setattr(expert_parlays, "load_player_week_stats", _not_ingested)
    with pytest.raises(ExpertPickError, match="No ingested player_stats"):
        publish_expert_parlay(LEGS, 180, now=BEFORE_KICKOFF)


def test_a_name_that_matches_two_players_is_refused_not_guessed(seeded, monkeypatch):
    frame = pl.concat([stats_frame(), pl.DataFrame([_row("00-other", "Matthew Stafford", "WR", "NYG", 1)], schema=_SCHEMA)])
    monkeypatch.setattr(expert_parlays, "load_player_week_stats", lambda season: frame)
    with pytest.raises(ExpertPickError, match="more than one player"):
        publish_expert_parlay(LEGS, 180, now=BEFORE_KICKOFF)


def test_the_same_parlay_can_never_be_published_twice(seeded):
    publish_expert_parlay(LEGS, 180, now=BEFORE_KICKOFF)
    with pytest.raises(ExpertPickError, match="already published"):
        publish_expert_parlay(LEGS, 190, now=BEFORE_KICKOFF)  # even at a different price
    assert len(read_current_picks()) == 1


def test_dry_run_build_writes_nothing(seeded):
    build_expert_parlay(LEGS, 180, now=BEFORE_KICKOFF)
    assert read_current_picks() == []


# ---------------------------------------------------------------- grading

def _publish_default():
    return publish_expert_parlay(LEGS, 180, now=BEFORE_KICKOFF)


def test_every_leg_winning_settles_the_parlay_as_a_win_at_the_parlay_price(seeded, monkeypatch):
    pick = _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)

    result = _settle(monkeypatch, stats_frame(week2=True, stafford=248, skattebo=55))

    assert result["settled"] == [{"pick_id": pick.pick_id, "category": "EXPERT_PARLAYS", "result": "WIN"}]
    settled = read_current_picks()[0]
    assert [lr["result"] for lr in settled["leg_results"]] == ["WIN", "WIN", "WIN"]
    assert settled["leg_results"][1]["detail"] == "248 passing yards"
    record = compute_category_record(read_current_picks(), "EXPERT_PARLAYS")
    assert (record.wins, record.losses) == (1, 0)
    assert record.total_units == pytest.approx(1.8)  # +180 on 1 unit


def test_one_losing_leg_loses_the_parlay_and_says_which(seeded, monkeypatch):
    _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)

    result = _settle(monkeypatch, stats_frame(week2=True, stafford=248, skattebo=39))  # 39 < 40

    assert result["settled"][0]["result"] == "LOSS"
    leg_results = read_current_picks()[0]["leg_results"]
    assert [lr["result"] for lr in leg_results] == ["WIN", "WIN", "LOSS"]
    assert compute_category_record(read_current_picks(), "EXPERT_PARLAYS").total_units == pytest.approx(-1.0)


def test_the_exact_threshold_counts_for_an_at_least_leg(seeded, monkeypatch):
    """'210+' means 210 wins: over 209.5, so exactly 210 is a WIN and 209 a LOSS."""
    _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)
    result = _settle(monkeypatch, stats_frame(week2=True, stafford=210, skattebo=40))
    assert result["settled"][0]["result"] == "WIN"


def test_a_lost_leg_settles_the_parlay_even_if_another_games_leg_hasnt_been_played(seeded, monkeypatch):
    legs = [LEGS[0], {"type": "moneyline", "game_id": OTHER_GAME_ID, "team": "LA", "price": -110}]
    publish_expert_parlay(legs, 250, now=BEFORE_KICKOFF)
    _finish(GAME_ID, home_score=10, away_score=27)  # LA loses; the Week 3 game is still scheduled

    result = _settle(monkeypatch, stats_frame())

    assert result["settled"][0]["result"] == "LOSS"
    assert [lr["result"] for lr in read_current_picks()[0]["leg_results"]] == ["LOSS", "NOT_GRADED"]


def test_a_parlay_stays_pending_while_any_leg_is_unplayed(seeded, monkeypatch):
    legs = [LEGS[0], {"type": "moneyline", "game_id": OTHER_GAME_ID, "team": "LA", "price": -110}]
    pick = publish_expert_parlay(legs, 250, now=BEFORE_KICKOFF)
    _finish(GAME_ID, home_score=27, away_score=10)  # leg 1 wins, leg 2 not played

    result = _settle(monkeypatch, stats_frame())

    assert result["settled"] == [] and result["still_pending"] == [pick.pick_id]
    assert read_current_picks()[0]["status"] == "PUBLISHED"


def test_a_stale_box_score_keeps_the_parlay_pending_instead_of_grading_a_leg_wrong(seeded, monkeypatch):
    """The game is final but the ingested player_stats snapshot has no Week 2 rows yet -
    that is 'not ingested yet', never 'the player got 0 yards'."""
    pick = _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)

    result = _settle(monkeypatch, stats_frame(week2=False))

    assert result["settled"] == [] and result["still_pending"] == [pick.pick_id]


def test_a_player_with_no_stat_row_goes_to_human_review_not_a_guess(seeded, monkeypatch):
    pick = _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)

    result = _settle(monkeypatch, stats_frame(week2=True, stafford=248, skattebo_row=False))

    assert result["settled"] == []
    assert result["needs_review"][0]["pick_id"] == pick.pick_id
    assert "did he not play" in result["needs_review"][0]["reason"]
    assert read_current_picks()[0]["status"] == "PUBLISHED"


def test_a_pushed_leg_goes_to_human_review_since_the_book_reprices_the_parlay(seeded, monkeypatch):
    legs = [LEGS[0], {"type": "player_prop", "game_id": GAME_ID, "player": "Matthew Stafford", "stat": "passing_yards",
                      "direction": "over", "line": 250}]
    pick = publish_expert_parlay(legs, 200, now=BEFORE_KICKOFF)
    _finish(GAME_ID, home_score=27, away_score=10)

    result = _settle(monkeypatch, stats_frame(week2=True, stafford=250, skattebo=0))

    assert result["settled"] == []
    assert result["needs_review"][0]["pick_id"] == pick.pick_id


def test_a_parlay_never_leaks_into_the_single_pick_or_model_records(seeded, monkeypatch):
    _publish_default()
    _finish(GAME_ID, home_score=27, away_score=10)
    _settle(monkeypatch, stats_frame(week2=True, stafford=248, skattebo=55))

    picks = read_current_picks()
    for category in ("EXPERT_PICKS", "BEST_BETS", "ALL_MODEL_PREDICTIONS"):
        assert compute_category_record(picks, category).n_settled == 0
    assert compute_category_record(picks, "EXPERT_PARLAYS").n_settled == 1
