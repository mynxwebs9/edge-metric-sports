"""Best Bets must be published on the VALUE side - the side the model likes more than the
market does - not the model's predicted winner. A real bug this guards against: two Week 2
Best Bets were published on the home favorite because it was the model's predicted winner,
even though the model rated that favorite LOWER than the market did (value was on the
underdog). These tests also cover the one-Best-Bet-per-game tie-break, idempotency, and the
never-after-kickoff rule."""

from __future__ import annotations

import pytest

from nfl_predict.api import reconstruction as recon
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.decision.best_bets_publishing import build_best_bet, publish_best_bets_for_week
from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.pick_ledger import read_current_picks
from nfl_predict.decision.schemas import Decision, DecisionRecord, ReasonCode
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.live.prediction_publication import PublicModelPrediction, PublicationState, publish_model_prediction
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider

GAME_ID = "2026_03_CLE_TB"
SEASON, WEEK = 2026, 3
BEFORE_KICKOFF = "2026-09-25T12:00:00+00:00"  # kickoff 2026-09-27 13:00 ET
AFTER_KICKOFF = "2026-09-28T12:00:00+00:00"


@pytest.fixture
def seeded(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [("4800", "TB", "Tampa Bay Buccaneers", "Buccaneers"), ("1050", "CLE", "Cleveland Browns", "Browns")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES (?, ?, 'REG', ?, '2026-09-27', '2026-09-27T13:00:00', '4800', '1050', 'TB', 'CLE', 'scheduled', '2026-09-24T00:00:00')",
        (GAME_ID, SEASON, WEEK),
    )
    conn.commit()
    conn.close()
    return isolated_data_dir


def _seed_model_and_market(model_home_prob: float, model_margin: float, home_spread: float = -8.5, home_ml: int = -395, away_ml: int = 310):
    publish_model_prediction(
        PublicModelPrediction(
            prediction_id="2026-09-24T08:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
            generated_at="2026-09-24T08:00:00+00:00", elo_home_win_probability=model_home_prob, elo_predicted_margin=model_margin,
            ridge_predicted_margin=model_margin, lightgbm_predicted_margin=model_margin, model_agreement_all_agree=True,
            model_agreement_dispersion=1.0, predicted_winner="home" if model_home_prob >= 0.5 else "away",
        ),
        PublicationState.PUBLISHED,
    )
    events = [OddsEvent(provider_event_id="evt_cletb", game_id=None, home_team="Tampa Bay Buccaneers", away_team="Cleveland Browns", commence_time="2026-09-27T17:00:00Z")]
    snap = OddsMarketSnapshot(
        provider_event_id="evt_cletb", fetched_at="2026-09-24T09:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread, home_spread_price=-112, away_spread_price=-108,
        home_moneyline=home_ml, away_moneyline=away_ml, total_line=44.5, over_price=-110, under_price=-110,
    )

    class _FakeOdds(OddsProvider):
        def get_events(self):
            return events

        def get_markets_for_sport(self):
            return [snap]

    fetch_and_snapshot_live_odds(_FakeOdds(), season=SEASON)


def _decide(market_type: str, decision: Decision):
    market = recon.latest_market_point(SEASON, WEEK, GAME_ID)
    append_decision_record(DecisionRecord(
        decision_id=f"{GAME_ID}_{market_type}_2026-09-24T10:00:00+00:00", game_id=GAME_ID,
        decision_timestamp="2026-09-24T10:00:00+00:00", kickoff_timestamp="2026-09-27T13:00:00", decision=decision,
        reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1", market_type=market_type,
        input_packet_hash="h", validation_status="PROSPECTIVE", market_provider_event_id=market.provider_event_id,
        market_snapshot_reference=market.market_snapshot_reference, market_snapshot_timestamp=market.snapshot_timestamp,
        research_id="research_1",
    ), SEASON, WEEK)


def test_a_model_favorite_the_market_rates_higher_is_bet_against_on_the_underdog(seeded):
    """The real bug: the model favors home (62%) but the market prices home at ~77%, so the
    model thinks the home favorite is OVERPRICED - the value, and the Best Bet, is the away
    underdog at its own price, not the home favorite the model 'predicts' to win."""
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("moneyline", Decision.QUALIFIED_BET)

    pick, detail = build_best_bet(GAME_ID, SEASON, WEEK, "2026-09-27T17:00:00+00:00", BEFORE_KICKOFF)

    assert (pick.market_type, pick.selection, pick.price) == ("moneyline", "away", 310)
    assert detail["disagreement"] < 0
    assert pick.model_prediction_snapshot["predicted_winner"] == "home"  # the very thing NOT used to pick the side


def test_a_model_that_likes_the_home_team_more_than_the_market_bets_the_home_team(seeded):
    _seed_model_and_market(model_home_prob=0.90, model_margin=12.0)
    _decide("moneyline", Decision.QUALIFIED_BET)

    pick, _ = build_best_bet(GAME_ID, SEASON, WEEK, "2026-09-27T17:00:00+00:00", BEFORE_KICKOFF)

    assert (pick.selection, pick.price) == ("home", -395)


def test_a_spread_best_bet_is_the_side_the_model_margin_favors_with_the_home_spread_as_the_line(seeded):
    # Market: home -8.5 (implied home margin +8.5). Model margin +3.6 -> model likes the AWAY side by 4.9.
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("spread", Decision.QUALIFIED_BET)

    pick, _ = build_best_bet(GAME_ID, SEASON, WEEK, "2026-09-27T17:00:00+00:00", BEFORE_KICKOFF)

    assert (pick.market_type, pick.selection, pick.line, pick.price) == ("spread", "away", -8.5, -108)


def test_when_both_markets_qualify_the_larger_multiple_of_its_own_threshold_wins(seeded):
    # spread gap 4.9 / 3.0 = 1.6x; moneyline gap ~0.157 / 0.03 = 5.2x -> moneyline wins.
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("spread", Decision.QUALIFIED_BET)
    _decide("moneyline", Decision.QUALIFIED_BET)

    pick, detail = build_best_bet(GAME_ID, SEASON, WEEK, "2026-09-27T17:00:00+00:00", BEFORE_KICKOFF)

    assert pick.market_type == "moneyline"
    assert detail["qualified_markets"] == ["spread", "moneyline"]


def test_only_a_qualified_bet_is_ever_published(seeded):
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("moneyline", Decision.WATCH)
    _decide("spread", Decision.LEAN)

    assert build_best_bet(GAME_ID, SEASON, WEEK, "2026-09-27T17:00:00+00:00", BEFORE_KICKOFF) is None
    result = publish_best_bets_for_week(WEEK, now=BEFORE_KICKOFF)
    assert result["published"] == [] and result["not_qualified"] == 1
    assert read_current_picks() == []


def test_publishing_records_the_decision_provenance_and_is_idempotent(seeded):
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("moneyline", Decision.QUALIFIED_BET)

    first = publish_best_bets_for_week(WEEK, now=BEFORE_KICKOFF)
    assert [(p["game_id"], p["selection"], p["price"]) for p in first["published"]] == [(GAME_ID, "away", 310)]

    stored = read_current_picks()[0]
    assert stored["category"] == "BEST_BETS" and stored["pick_id"] == f"{GAME_ID}_best_bet"
    assert stored["decision_id"] == f"{GAME_ID}_moneyline_2026-09-24T10:00:00+00:00"
    assert stored["research_snapshot_id"] == "research_1"
    assert stored["market_snapshot_id"] == recon.latest_market_point(SEASON, WEEK, GAME_ID).market_snapshot_reference

    second = publish_best_bets_for_week(WEEK, now=BEFORE_KICKOFF)
    assert second["published"] == [] and second["already_published"] == [GAME_ID]
    assert len(read_current_picks()) == 1


def test_dry_run_reports_the_pick_but_writes_nothing(seeded):
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("moneyline", Decision.QUALIFIED_BET)

    result = publish_best_bets_for_week(WEEK, dry_run=True, now=BEFORE_KICKOFF)

    assert result["published"][0]["selection"] == "away"
    assert read_current_picks() == []


def test_a_qualified_bet_is_never_published_after_kickoff(seeded):
    _seed_model_and_market(model_home_prob=0.618, model_margin=3.6)
    _decide("moneyline", Decision.QUALIFIED_BET)

    result = publish_best_bets_for_week(WEEK, now=AFTER_KICKOFF)

    assert result["published"] == []
    assert result["too_late"][0]["game_id"] == GAME_ID
    assert read_current_picks() == []
