"""Real settlement pass: grades every PUBLISHED pick (in either PickCategory) against the
real final score, using the same deterministic spread/moneyline logic Phase 7 already built
and validated (nfl_predict.decision.settlement) - this module only wires that pure
computation to the real games table and the ledger's own settle_pick, never invents a new
grading rule."""

from __future__ import annotations

from datetime import datetime, timezone

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.decision.pick_ledger import PickCategory, PublishedPick, publish_pick, read_current_picks
from nfl_predict.decision.pick_settlement import settle_all_pending_picks

SEASON, WEEK = 2026, 2


def _seed_game(game_id: str, home_id: str, away_id: str, home_score: int | None, away_score: int | None, status: str):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT OR REPLACE INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [(home_id, home_id, home_id, home_id), (away_id, away_id, away_id, away_id)],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, home_score, away_score, "
        "game_status, normalized_at) VALUES (?, ?, 'REG', ?, '2026-09-20', '2026-09-20T16:05:00', "
        "?, ?, ?, ?, ?, ?, ?, '2026-09-16T00:00:00')",
        (game_id, SEASON, WEEK, home_id, away_id, home_id, away_id, home_score, away_score, status),
    )
    conn.commit()
    conn.close()


def _publish(game_id, market_type, selection, line, price, category=PickCategory.ALL_MODEL_PREDICTIONS):
    publish_pick(PublishedPick(
        pick_id=f"{game_id}_{market_type}_{category.value}", game_id=game_id, published_at="2026-09-16T12:00:00+00:00",
        kickoff_at="2026-09-20T16:05:00+00:00", decision_id="d1", rule_version="v1", category=category.value,
        market_type=market_type, selection=selection, line=line, price=price, sportsbook_or_source="test",
        market_snapshot_id="snap1", model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))


def test_a_moneyline_pick_on_the_winning_side_settles_as_a_win(isolated_data_dir):
    _seed_game("g1", "HOME", "AWAY", home_score=27, away_score=20, status="final")
    _publish("g1", "moneyline", "home", None, -150)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"] == [{"pick_id": "g1_moneyline_ALL_MODEL_PREDICTIONS", "category": "ALL_MODEL_PREDICTIONS", "result": "WIN"}]

    picks = {p["pick_id"]: p for p in read_current_picks()}
    assert picks["g1_moneyline_ALL_MODEL_PREDICTIONS"]["status"] == "SETTLED"
    assert picks["g1_moneyline_ALL_MODEL_PREDICTIONS"]["settlement"] == "WIN"


def test_a_moneyline_pick_on_the_losing_side_settles_as_a_loss(isolated_data_dir):
    _seed_game("g2", "HOME", "AWAY", home_score=10, away_score=24, status="final")
    _publish("g2", "moneyline", "home", None, -150)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"][0]["result"] == "LOSS"


def test_a_spread_pick_that_covers_settles_as_a_win(isolated_data_dir):
    # home wins by 10, home was favored by -3.0 (line is the home spread) -> covers
    _seed_game("g3", "HOME", "AWAY", home_score=24, away_score=14, status="final")
    _publish("g3", "spread", "home", -3.0, -110)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"][0]["result"] == "WIN"


def test_a_spread_pick_that_lands_exactly_on_the_number_is_a_push(isolated_data_dir):
    # home wins by exactly 3, line is home -3.0 -> push
    _seed_game("g4", "HOME", "AWAY", home_score=23, away_score=20, status="final")
    _publish("g4", "spread", "home", -3.0, -110)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"][0]["result"] == "PUSH"


def test_a_pick_whose_game_has_not_finished_yet_stays_pending_not_settled(isolated_data_dir):
    _seed_game("g5", "HOME", "AWAY", home_score=None, away_score=None, status="scheduled")
    _publish("g5", "moneyline", "away", None, 130)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"] == []
    assert result["still_pending"] == ["g5_moneyline_ALL_MODEL_PREDICTIONS"]


def test_an_already_settled_pick_is_never_re_graded(isolated_data_dir):
    _seed_game("g6", "HOME", "AWAY", home_score=27, away_score=20, status="final")
    _publish("g6", "moneyline", "home", None, -150)
    first = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert len(first["settled"]) == 1

    second = settle_all_pending_picks(now="2026-09-22T00:00:00+00:00")
    assert second["settled"] == []
    assert second["still_pending"] == []


def test_best_bets_and_all_model_predictions_settle_independently_in_one_pass(isolated_data_dir):
    _seed_game("g7", "HOME", "AWAY", home_score=27, away_score=20, status="final")
    _publish("g7", "moneyline", "home", None, -150, category=PickCategory.BEST_BETS)
    _publish("g7", "spread", "home", -3.0, -110, category=PickCategory.ALL_MODEL_PREDICTIONS)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    categories = {s["category"] for s in result["settled"]}
    assert categories == {"BEST_BETS", "ALL_MODEL_PREDICTIONS"}
    assert len(result["settled"]) == 2
