"""ALL_MODEL_PREDICTIONS was designed (PickCategory enum, /api/nfl/performance's
all_model_predictions_record) but never actually published to by any code path before this
module - these tests prove the new publishing function actually works end-to-end against
real, persisted model/market data, and that it is a genuinely separate record from
BEST_BETS (never conflated, never gated by the decision engine)."""

from __future__ import annotations

import pytest

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.decision.pick_ledger import PickCategory, read_current_picks
from nfl_predict.decision.pick_publishing import (
    build_all_model_predictions_pick,
    publish_all_model_predictions_for_week,
)
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.live.prediction_publication import PublicModelPrediction, PublicationState, publish_model_prediction
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider

GAME_ID = "2026_02_JAX_DEN"
SEASON, WEEK = 2026, 2
HOME_ID, AWAY_ID = "2250", "1400"  # JAX, DEN


@pytest.fixture
def seeded_game(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [("2250", "JAX", "Jacksonville Jaguars", "Jaguars"), ("1400", "DEN", "Denver Broncos", "Broncos")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES (?, ?, 'REG', ?, '2026-09-20', '2026-09-20T16:05:00', ?, ?, 'JAX', 'DEN', 'scheduled', '2026-09-16T00:00:00')",
        (GAME_ID, SEASON, WEEK, HOME_ID, AWAY_ID),
    )
    conn.commit()
    conn.close()
    return isolated_data_dir


def _publish_real_model_and_market(predicted_winner="away"):
    publish_model_prediction(
        PublicModelPrediction(
            prediction_id="2026-09-16T08:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
            generated_at="2026-09-16T08:00:00+00:00", elo_home_win_probability=0.41, elo_predicted_margin=-2.9,
            ridge_predicted_margin=-2.5, lightgbm_predicted_margin=-1.1, model_agreement_all_agree=True,
            model_agreement_dispersion=1.8, predicted_winner=predicted_winner,
        ),
        PublicationState.PUBLISHED,
    )
    events = [OddsEvent(provider_event_id="evt_jaxden", game_id=None, home_team="Jacksonville Jaguars", away_team="Denver Broncos", commence_time="2026-09-20T20:05:00Z")]
    snap = OddsMarketSnapshot(
        provider_event_id="evt_jaxden", fetched_at="2026-09-16T09:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=2.5, away_spread_traditional=-2.5, home_spread_price=-110, away_spread_price=-110,
        home_moneyline=130, away_moneyline=-150, total_line=44.5, over_price=-110, under_price=-110,
    )

    class _FakeOdds(OddsProvider):
        def get_events(self):
            return events

        def get_markets_for_sport(self):
            return [snap]

    fetch_and_snapshot_live_odds(_FakeOdds(), season=SEASON)


def test_returns_none_when_no_model_prediction_exists_yet(seeded_game):
    assert build_all_model_predictions_pick(GAME_ID, SEASON, WEEK) is None


def test_returns_none_when_model_exists_but_market_does_not(seeded_game):
    publish_model_prediction(
        PublicModelPrediction(
            prediction_id="p1", game_id=GAME_ID, season=SEASON, week=WEEK, generated_at="2026-09-16T08:00:00+00:00",
            elo_home_win_probability=0.41, elo_predicted_margin=-2.9, ridge_predicted_margin=None,
            lightgbm_predicted_margin=None, model_agreement_all_agree=None, model_agreement_dispersion=None,
            predicted_winner="away",
        ),
        PublicationState.PUBLISHED,
    )
    assert build_all_model_predictions_pick(GAME_ID, SEASON, WEEK) is None


def test_builds_a_real_priced_pick_matching_the_models_predicted_winner(seeded_game):
    _publish_real_model_and_market(predicted_winner="away")
    pick = build_all_model_predictions_pick(GAME_ID, SEASON, WEEK)
    assert pick is not None
    assert pick.category == PickCategory.ALL_MODEL_PREDICTIONS.value
    assert pick.market_type == "moneyline"
    assert pick.selection == "away"  # model favors DEN (away)
    assert pick.price == -150  # DEN's real away_moneyline, not the home price
    assert pick.line is None
    assert pick.pick_id == f"{GAME_ID}_all_model_predictions"


def test_selection_flips_to_home_when_the_model_favors_the_home_team(seeded_game):
    _publish_real_model_and_market(predicted_winner="home")
    pick = build_all_model_predictions_pick(GAME_ID, SEASON, WEEK)
    assert pick.selection == "home"
    assert pick.price == 130  # JAX's real home_moneyline


def test_publish_all_model_predictions_for_week_publishes_and_records_it_separately_from_best_bets(seeded_game):
    _publish_real_model_and_market(predicted_winner="away")
    result = publish_all_model_predictions_for_week(SEASON, WEEK)
    assert result["published"] == [GAME_ID]
    assert result["already_published"] == []

    picks = read_current_picks()
    assert len(picks) == 1
    assert picks[0]["category"] == "ALL_MODEL_PREDICTIONS"
    assert picks[0]["status"] == "PUBLISHED"


def test_publish_all_model_predictions_for_week_is_idempotent_on_a_second_run(seeded_game):
    _publish_real_model_and_market(predicted_winner="away")
    publish_all_model_predictions_for_week(SEASON, WEEK)
    result = publish_all_model_predictions_for_week(SEASON, WEEK)
    assert result["published"] == []
    assert result["already_published"] == [GAME_ID]
    assert len(read_current_picks()) == 1  # still exactly one pick, never duplicated


def test_a_game_with_no_data_yet_is_reported_not_silently_skipped(seeded_game):
    result = publish_all_model_predictions_for_week(SEASON, WEEK)
    assert result["no_data_yet"] == [GAME_ID]
    assert result["published"] == []
