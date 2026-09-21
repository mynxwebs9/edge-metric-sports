"""Phase 8B: FastAPI route-level tests via TestClient - no live network call anywhere (odds/
research provider status checks only ever construct a provider object, never call it; the
autouse `tests/conftest.py` fixture also strips any live credential env vars by default)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nfl_predict.api.main import app
from nfl_predict.content.prediction_writer import FixtureContentWriterProvider, write_prediction_preview
from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.pick_ledger import PublishedPick, SettlementResult, publish_pick, settle_pick
from nfl_predict.decision.schemas import Decision, DecisionRecord, ReasonCode
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.live.prediction_publication import PublicModelPrediction, PublicationState, publish_model_prediction
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider
from nfl_predict.research.input_packet import MarketContext, ModelPrediction, ResearchInputPacket
from nfl_predict.research.llm_provider import FixtureLLMProvider
from nfl_predict.research.run_research import run_research_for_game

CONTENT_FIXTURE = Path(__file__).resolve().parents[1] / "content" / "fixtures" / "content_fixture.json"

GAME_ID = "2026_01_DEN_KC"
SEASON, WEEK = 2026, 1
LLM_FIXTURE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "nfl_predict-api", "schema_version": "1"}


def test_current_slate_renders_a_real_scheduled_game(api_data_dir):
    resp = client.get("/api/nfl/slate/current")
    assert resp.status_code == 200
    body = resp.json()
    assert body["season"] == SEASON
    assert body["week"] == WEEK
    game_ids = [g["game_id"] for g in body["games"]]
    assert GAME_ID in game_ids
    card = next(g for g in body["games"] if g["game_id"] == GAME_ID)
    assert card["away_team"]["abbr"] == "DEN"
    assert card["home_team"]["abbr"] == "KC"
    assert card["model"]["available"] is False  # nothing published yet in this isolated test env
    assert card["market"]["available"] is False
    assert card["decision"]["available"] is False
    assert card["research"]["available"] is False


def test_game_detail_404s_for_an_unknown_game(api_data_dir):
    resp = client.get("/api/nfl/games/not_a_real_game")
    assert resp.status_code == 404


def test_schedule_weeks_lists_the_real_season_weeks(api_data_dir):
    resp = client.get("/api/nfl/schedule")
    assert resp.status_code == 200
    body = resp.json()
    assert body["season"] == SEASON
    assert WEEK in body["weeks"]
    assert body["current_week"] == WEEK


def test_schedule_for_a_specific_week_renders_the_same_shape_as_current_slate(api_data_dir):
    resp = client.get(f"/api/nfl/schedule/{WEEK}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["week"] == WEEK
    game_ids = [g["game_id"] for g in body["games"]]
    assert GAME_ID in game_ids


def test_schedule_for_a_week_with_no_games_404s_not_a_fabricated_empty_slate(api_data_dir):
    """A week that genuinely doesn't exist in the schedule (never ingested/scheduled) must
    404, never silently return an empty-but-valid-looking slate that could be confused with
    a real, fully-final week with zero games."""
    resp = client.get("/api/nfl/schedule/99")
    assert resp.status_code == 404


def test_game_detail_renders_real_model_market_research_and_decision_data(api_data_dir):
    publish_model_prediction(
        PublicModelPrediction(
            prediction_id="2026-09-12T08:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
            generated_at="2026-09-12T08:00:00+00:00", elo_home_win_probability=0.39, elo_predicted_margin=-3.2,
            ridge_predicted_margin=-3.4, lightgbm_predicted_margin=-0.9, model_agreement_all_agree=True,
            model_agreement_dispersion=2.5, predicted_winner="away",
        ),
        PublicationState.PUBLISHED,
    )

    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    snap = OddsMarketSnapshot(
        provider_event_id="evt_denkc", fetched_at="2026-09-14T17:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=-2.5, away_spread_traditional=2.5, home_spread_price=-110, away_spread_price=-110,
        home_moneyline=-142, away_moneyline=120, total_line=44.5, over_price=-110, under_price=-110,
    )

    class _FakeOdds(OddsProvider):
        def get_events(self):
            return events

        def get_markets_for_sport(self):
            return [snap]

    fetch_and_snapshot_live_odds(_FakeOdds(), season=SEASON)

    packet = ResearchInputPacket(
        game_id=GAME_ID, season=SEASON, week=WEEK, season_type="REG",
        home_team_id="2310", away_team_id="1400", kickoff_timestamp="2026-09-14T20:15:00+00:00",
        packet_generated_at="2026-09-12T09:00:00+00:00",
        elo=ModelPrediction(model_id="elo_v2", available=True, predicted_margin=-3.2, home_win_probability=0.39),
        ridge=ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="test"),
        lightgbm=ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason="test"),
        market=MarketContext(available=False, unavailable_reason="test"),
        model_market_disagreement_points=None, known_qb_continuity_note=None,
        known_personnel_continuity_note=None, known_injury_summary=None,
    )
    run_research_for_game(
        packet=packet, provider=FixtureLLMProvider(LLM_FIXTURE), research_prompt_template_text="x",
        run_id="2026-09-12T09-00-00.000000+00-00", now="2026-09-12T09:00:00+00:00",
    )

    record = DecisionRecord(
        decision_id="d1", game_id=GAME_ID, decision_timestamp="2026-09-12T09:30:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.QUALIFIED_BET, reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h", validation_status="PROSPECTIVE",
    )
    append_decision_record(record, SEASON, WEEK)

    resp = client.get(f"/api/nfl/games/{GAME_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]["available"] is True
    assert body["model"]["elo_predicted_margin"] == -3.2
    assert body["market"]["available"] is True
    assert body["market"]["home_spread_traditional"] == -2.5
    assert body["research"]["available"] is True
    assert body["research"]["classification"] == "NO_MATERIAL_NEW_INFORMATION"
    assert body["spread_decision"]["decision"] == "QUALIFIED_BET"
    assert body["spread_decision"]["decision_label"] == "Qualified Bet"
    # A QUALIFIED_BET decision is real, but must never be confused with an actual published
    # pick - no pick was published in this test, so this must be False even though the
    # decision itself qualified (the real bug a screenshot caught: both market types showed
    # "BEST BET" even though only one was ever actually published).
    assert body["spread_decision"]["is_published_best_bet"] is False
    assert body["spread_decision"]["reason_codes"][0]["code"] == "MODEL_MARKET_DISAGREEMENT"
    assert body["moneyline_decision"]["available"] is False
    # -3.2 - (-(-2.5)) = -3.2 - 2.5 = -5.7
    assert body["model_market_disagreement_points"] == -5.7
    assert body["preview"]["available"] is False  # no article generated yet in this test
    assert body["system_pick"]["available"] is False  # no ALL_MODEL_PREDICTIONS pick published in this test


def test_system_pick_shows_the_real_published_all_model_predictions_pick_everywhere(api_data_dir):
    """system_pick must show up on BOTH the game-detail response and the slate/schedule
    card for the same game - and it's a completely separate thing from the Best Bets
    decision status, published for every game with real model/market data regardless of
    whether that game ever reaches QUALIFIED_BET."""
    publish_model_prediction(
        PublicModelPrediction(
            prediction_id="2026-09-12T08:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
            generated_at="2026-09-12T08:00:00+00:00", elo_home_win_probability=0.39, elo_predicted_margin=-3.2,
            ridge_predicted_margin=-3.4, lightgbm_predicted_margin=-0.9, model_agreement_all_agree=True,
            model_agreement_dispersion=2.5, predicted_winner="away",
        ),
        PublicationState.PUBLISHED,
    )
    publish_pick(PublishedPick(
        pick_id=f"{GAME_ID}_all_model_predictions", game_id=GAME_ID, published_at="2026-09-12T08:05:00+00:00",
        kickoff_at="2026-09-14T20:15:00+00:00", decision_id="d1", rule_version="v1",
        category="ALL_MODEL_PREDICTIONS", market_type="moneyline", selection="away", line=None, price=120,
        sportsbook_or_source="test", market_snapshot_id="snap1", model_prediction_snapshot={}, research_snapshot_id=None,
        validation_status="PROSPECTIVE",
    ))

    detail = client.get(f"/api/nfl/games/{GAME_ID}").json()
    assert detail["system_pick"]["available"] is True
    assert detail["system_pick"]["selection"] == "away"
    assert detail["system_pick"]["selection_team"]["abbr"] == "DEN"
    assert detail["system_pick"]["price"] == 120
    assert detail["system_pick"]["market_type"] == "moneyline"
    assert detail["system_pick"]["status"] == "PUBLISHED"

    slate = client.get("/api/nfl/slate/current").json()
    card = next(g for g in slate["games"] if g["game_id"] == GAME_ID)
    assert card["system_pick"]["available"] is True
    assert card["system_pick"]["selection_team"]["abbr"] == "DEN"


def test_game_ids_endpoint_lists_every_real_scheduled_game_cheaply(api_data_dir):
    """Built for the website's sitemap - must return every real game_id across the whole
    season without doing any of the heavy per-game model/market/research/decision
    reconstruction `/api/nfl/schedule/{week}` does (that's what made sitemap generation slow
    enough to time out in production)."""
    resp = client.get("/api/nfl/game-ids")
    assert resp.status_code == 200
    body = resp.json()
    assert body["season"] == SEASON
    ids = [g["game_id"] for g in body["game_ids"]]
    assert GAME_ID in ids
    entry = next(g for g in body["game_ids"] if g["game_id"] == GAME_ID)
    assert entry["week"] == WEEK


def test_game_card_headline_decision_matches_the_moneyline_not_the_spread(api_data_dir):
    """Real bug caught by a screenshot: a game card showed a "No Bet" headline badge right
    next to a "Best Bet" pill for the same game. `system_pick` (what the pill is about) is
    always a moneyline pick - the card's headline decision must track moneyline too, or the
    two can legitimately disagree (spread NO_BET, moneyline QUALIFIED_BET) and look
    self-contradictory."""
    append_decision_record(DecisionRecord(
        decision_id="d_spread", game_id=GAME_ID, decision_timestamp="2026-09-12T09:30:00+00:00",
        kickoff_timestamp="2026-09-14T20:15:00", decision=Decision.NO_BET,
        reason_codes=(ReasonCode.DISAGREEMENT_BELOW_MINIMUM,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h1", validation_status="PROSPECTIVE",
    ), SEASON, WEEK)
    append_decision_record(DecisionRecord(
        decision_id="d_ml", game_id=GAME_ID, decision_timestamp="2026-09-12T09:30:00+00:00",
        kickoff_timestamp="2026-09-14T20:15:00", decision=Decision.QUALIFIED_BET,
        reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="moneyline", input_packet_hash="h2", validation_status="PROSPECTIVE",
    ), SEASON, WEEK)

    slate = client.get("/api/nfl/slate/current").json()
    card = next(g for g in slate["games"] if g["game_id"] == GAME_ID)
    assert card["decision"]["market_type"] == "moneyline"
    assert card["decision"]["decision"] == "QUALIFIED_BET"


def _expert_pick(market_type: str, selection: str, line, price, note=None) -> PublishedPick:
    return PublishedPick(
        pick_id=f"{GAME_ID}_expert_{market_type}", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00",
        kickoff_at="2026-09-14T20:15:00+00:00", decision_id="n/a - expert pick", rule_version="n/a",
        category="EXPERT_PICKS", market_type=market_type, selection=selection, line=line, price=price,
        sportsbook_or_source="Expert-supplied line and price", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE", note=note,
    )


def test_expert_picks_endpoint_has_an_honest_empty_state(api_data_dir):
    body = client.get("/api/nfl/expert-picks").json()
    assert body["open_picks"] == []
    assert body["settled_picks"] == []
    assert body["record"]["category"] == "EXPERT_PICKS"
    assert body["record"]["n_settled"] == 0
    assert body["record"]["win_rate"] is None


def test_expert_picks_endpoint_separates_open_and_settled_and_never_leaks_into_model_records(api_data_dir):
    publish_pick(_expert_pick("spread", "away", -3.0, -110, note="Broncos +3 is a full field goal of cushion."))
    publish_pick(_expert_pick("moneyline", "away", None, 120))
    settle_pick(f"{GAME_ID}_expert_moneyline", SettlementResult.WIN, "2026-09-15T04:00:00+00:00", "test")

    body = client.get("/api/nfl/expert-picks").json()

    assert [p["pick_id"] for p in body["open_picks"]] == [f"{GAME_ID}_expert_spread"]
    open_pick = body["open_picks"][0]
    assert open_pick["selection_team"]["abbr"] == "DEN"
    assert open_pick["line"] == -3.0
    assert open_pick["note"] == "Broncos +3 is a full field goal of cushion."
    assert open_pick["settlement"] is None

    assert [p["pick_id"] for p in body["settled_picks"]] == [f"{GAME_ID}_expert_moneyline"]
    assert body["settled_picks"][0]["settlement"] == "WIN"

    assert (body["record"]["n_settled"], body["record"]["wins"], body["record"]["losses"]) == (1, 1, 0)
    assert body["record"]["total_units"] == 1.2  # +120 win at 1 unit

    # A human's picks are their own record - they must never appear in, or change, the model's.
    assert client.get("/api/nfl/best-bets").json()["picks"] == []
    performance = client.get("/api/nfl/performance").json()
    assert performance["best_bets_record"]["n_settled"] == 0
    assert performance["all_model_predictions_record"]["n_settled"] == 0


def test_only_the_actually_published_market_type_shows_is_published_best_bet(api_data_dir):
    """Regression test for a real screenshot-caught bug: when BOTH spread and moneyline
    reach QUALIFIED_BET but only ONE is actually published (the project's one-pick-per-game
    policy), the game page must distinguish them - never show both as "the" Best Bet."""
    spread_record = DecisionRecord(
        decision_id="d_spread", game_id=GAME_ID, decision_timestamp="2026-09-12T09:30:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.QUALIFIED_BET, reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h1", validation_status="PROSPECTIVE",
    )
    ml_record = DecisionRecord(
        decision_id="d_ml", game_id=GAME_ID, decision_timestamp="2026-09-12T09:30:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.QUALIFIED_BET, reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="moneyline", input_packet_hash="h2", validation_status="PROSPECTIVE",
    )
    append_decision_record(spread_record, SEASON, WEEK)
    append_decision_record(ml_record, SEASON, WEEK)

    # Only the moneyline pick actually gets published - mirrors the real DEN@KC tie-break.
    publish_pick(PublishedPick(
        pick_id="p_ml", game_id=GAME_ID, published_at="2026-09-12T10:00:00+00:00", kickoff_at="2026-09-14T20:15:00",
        decision_id="d_ml", rule_version="v1", category="BEST_BETS", market_type="moneyline", selection="away",
        line=None, price=114, sportsbook_or_source="consensus", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))

    body = client.get(f"/api/nfl/games/{GAME_ID}").json()
    assert body["spread_decision"]["decision"] == "QUALIFIED_BET"
    assert body["spread_decision"]["is_published_best_bet"] is False
    assert body["moneyline_decision"]["decision"] == "QUALIFIED_BET"
    assert body["moneyline_decision"]["is_published_best_bet"] is True


def test_game_detail_includes_a_real_generated_preview_article_when_one_exists(api_data_dir):
    write_prediction_preview(
        SEASON, WEEK, GAME_ID, "2310", "1400", "2026-09-14T20:15:00",
        provider=FixtureContentWriterProvider(CONTENT_FIXTURE),
        prompt_template_text=(Path(__file__).resolve().parents[2] / "prompts" / "prediction_writer_v2.md").read_text(encoding="utf-8"),
        run_id="test_preview_run", now="2026-09-12T12:00:00+00:00",
    )

    resp = client.get(f"/api/nfl/games/{GAME_ID}")
    body = resp.json()
    assert body["preview"]["available"] is True
    assert "Elo projects DEN by 3.2" in body["preview"]["text"]
    assert body["preview"]["prompt_version"] == "prediction_writer_v2"
    assert body["preview"]["model_provider"] == "fixture"


def test_best_bets_empty_state(api_data_dir):
    resp = client.get("/api/nfl/best-bets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["picks"] == []
    assert body["message"] == "No plays currently meet our qualification criteria."


def test_best_bets_returns_a_real_published_pick(api_data_dir):
    """Regression test: the API must resolve `selection` ("home"/"away", the internal
    settlement convention - see decision/settlement.py) into an actual team, never leave the
    frontend to display the raw enum string - a real bug found via manual testing with a
    genuinely published pick."""
    publish_pick(PublishedPick(
        pick_id="p1", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00", kickoff_at="2026-09-14T20:15:00",
        decision_id="d1", rule_version="v1", category="BEST_BETS", market_type="spread", selection="away",
        line=-2.5, price=-110, sportsbook_or_source="consensus", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))
    resp = client.get("/api/nfl/best-bets")
    body = resp.json()
    assert body["message"] is None
    assert len(body["picks"]) == 1
    pick = body["picks"][0]
    assert pick["pick_id"] == "p1"
    assert pick["selection"] == "away"
    assert pick["selection_team"]["abbr"] == "DEN"  # away team for 2026_01_DEN_KC
    assert pick["home_team"]["abbr"] == "KC"
    assert pick["away_team"]["abbr"] == "DEN"


def test_performance_reports_zeros_not_fabricated_data_when_ledger_is_empty(api_data_dir):
    resp = client.get("/api/nfl/performance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_bets_record"]["n_settled"] == 0
    assert body["best_bets_record"]["win_rate"] is None
    assert body["has_settled_history"] is False
    assert body["headline"] is None


def test_model_status_reports_real_rule_version_and_no_live_credentials(api_data_dir):
    resp = client.get("/api/nfl/model-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rule_set_version"]
    assert set(body["model_ids"]) == {"elo_v2", "ridge_margin_E_v1", "lightgbm_F_v1"}
    # tests/conftest.py's autouse fixture strips live credential env vars for every test.
    assert body["odds_provider_status"] == "ODDS_PROVIDER_UNAVAILABLE"
    assert body["research_provider_status"] == "RESEARCH_PROVIDER_UNAVAILABLE"


def test_decisions_endpoint_404s_for_an_unknown_game(api_data_dir):
    resp = client.get("/api/nfl/decisions/not_a_real_game")
    assert resp.status_code == 404


def test_decisions_endpoint_reconstructs_the_exact_referenced_market_not_a_newer_one(api_data_dir):
    """The item-6 lesson, exercised through the real HTTP endpoint: an OLD decision's market
    block must reflect the line that decision actually saw, even after a newer real odds
    refresh has since been persisted for the same provider_event_id."""
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]

    class _FakeOdds(OddsProvider):
        def __init__(self, snap):
            self._snap = snap

        def get_events(self):
            return events

        def get_markets_for_sport(self):
            return [self._snap]

    old_snap = OddsMarketSnapshot(
        provider_event_id="evt_denkc", fetched_at="2026-09-13T10:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=-1.0, away_spread_traditional=1.0, home_spread_price=-110, away_spread_price=-110,
        home_moneyline=-120, away_moneyline=100, total_line=44.5, over_price=-110, under_price=-110,
    )
    result = fetch_and_snapshot_live_odds(_FakeOdds(old_snap), season=SEASON)
    old_market = result.games[GAME_ID].market

    record = DecisionRecord(
        decision_id="d1", game_id=GAME_ID, decision_timestamp="2026-09-13T10:05:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.NO_BET, reason_codes=(ReasonCode.DISAGREEMENT_BELOW_MINIMUM,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h",
        market_provider_event_id=old_market.provider_event_id, market_snapshot_reference=old_market.market_snapshot_reference,
        market_snapshot_timestamp=old_market.snapshot_timestamp, validation_status="PROSPECTIVE",
    )
    append_decision_record(record, SEASON, WEEK)

    new_snap = OddsMarketSnapshot(
        provider_event_id="evt_denkc", fetched_at="2026-09-14T17:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=-3.0, away_spread_traditional=3.0, home_spread_price=-110, away_spread_price=-110,
        home_moneyline=-150, away_moneyline=130, total_line=44.5, over_price=-110, under_price=-110,
    )
    fetch_and_snapshot_live_odds(_FakeOdds(new_snap), season=SEASON)

    resp = client.get(f"/api/nfl/decisions/{GAME_ID}")
    body = resp.json()
    assert body["spread"]["market"]["available"] is True
    assert body["spread"]["market"]["home_spread_traditional"] == -1.0  # the OLD line the decision actually saw
    assert body["spread"]["market"]["snapshot_timestamp"] == "2026-09-13T10:00:00+00:00"
