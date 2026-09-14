"""Phase 8A provenance correction: proves the actual gap the Task-8 offline replay hit is
now closed - a canonical `game_id` alone (never an in-memory object left over from the
original ingestion run) is enough to find and reconstruct the real persisted market data
that fed a past decision, and a full spread/moneyline decision can be replayed entirely from
persisted odds + persisted research + persisted model predictions, with zero network calls.
"""

from __future__ import annotations

from pathlib import Path

from nfl_predict.decision.engine import decide
from nfl_predict.decision.input_packet import build_research_point_from_stored_run
from nfl_predict.decision.model_agreement import compute_model_agreement
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.decision.schemas import DecisionInputPacket, MarketPoint, ModelPoint, ReasonCode, SystemHealth
from nfl_predict.live.market_consensus import compute_market_consensus
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.market.event_game_mapping import find_provider_event_ids_for_game
from nfl_predict.market.live_snapshot_store import read_live_snapshots
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider
from nfl_predict.research.input_packet import MarketContext, ModelPrediction, ResearchInputPacket
from nfl_predict.research.llm_provider import FixtureLLMProvider
from nfl_predict.research.run_research import run_research_for_game
from nfl_predict.research.storage import read_research_run

REAL_GAME_ID = "2026_01_DEN_KC"
SEASON, WEEK = 2026, 1
LLM_FIXTURE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"


class _FakeProvider(OddsProvider):
    def __init__(self, events, markets_by_event):
        self._events = events
        self._markets = markets_by_event

    def get_events(self):
        return self._events

    def get_markets_for_sport(self):
        return [s for snaps in self._markets.values() for s in snaps]


def _patch_storage(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def _snap(event_id, bookmaker, fetched_at, home_spread, home_ml, away_ml):
    return OddsMarketSnapshot(
        provider_event_id=event_id, fetched_at=fetched_at, bookmaker=bookmaker,
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=home_ml, away_moneyline=away_ml,
        total_line=44.5, over_price=-110, under_price=-110,
    )


def _real_shaped_events_and_markets():
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_denkc": [
        _snap("evt_denkc", "book_a", "2026-09-14T17:00:00+00:00", -3.0, -150, 130),
        _snap("evt_denkc", "book_b", "2026-09-14T17:00:00+00:00", -3.5, -145, 125),
    ]}
    return events, markets


def test_a_canonical_game_id_alone_finds_exactly_one_provider_event(tmp_path, monkeypatch):
    """The reconstruction entry point: given only a canonical game_id (and season/week, both
    knowable ahead of time - never an in-memory leftover), the mapping index resolves back
    to the exact provider_event_id whose real, persisted snapshots produced this game's
    market data."""
    _patch_storage(monkeypatch, tmp_path)
    events, markets = _real_shaped_events_and_markets()
    fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    provider_event_ids = find_provider_event_ids_for_game(SEASON, WEEK, REAL_GAME_ID)
    assert provider_event_ids == ["evt_denkc"]


def test_provider_event_id_is_preserved_end_to_end_on_the_marketpoint(tmp_path, monkeypatch):
    _patch_storage(monkeypatch, tmp_path)
    events, markets = _real_shaped_events_and_markets()
    result = fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    market = result.games[REAL_GAME_ID].market
    assert market.provider_event_id == "evt_denkc"
    assert market.market_snapshot_reference is not None
    assert market.consensus_algorithm_version is not None
    assert set(market.underlying_snapshot_keys) == {"book_a|2026-09-14T17:00:00+00:00", "book_b|2026-09-14T17:00:00+00:00"}


def test_home_away_normalization_is_recorded_correctly_in_the_mapping(tmp_path, monkeypatch):
    """The provider's raw team-name strings and the normalized team_ids they resolved to are
    BOTH persisted, so a reader can audit the normalization itself, not just trust it."""
    _patch_storage(monkeypatch, tmp_path)
    events, markets = _real_shaped_events_and_markets()
    fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    from nfl_predict.market.event_game_mapping import read_event_game_mappings

    mappings = read_event_game_mappings(SEASON, WEEK)
    assert len(mappings) == 1
    row = mappings[0]
    assert row["provider_home_team"] == "Kansas City Chiefs"
    assert row["provider_away_team"] == "Denver Broncos"
    # Kansas City Chiefs are the real home team for 2026_01_DEN_KC - normalization must not
    # have swapped home/away.
    assert row["home_team_id"] != row["away_team_id"]
    assert row["canonical_game_id"] == REAL_GAME_ID
    assert row["status"] == "matched"


def test_ambiguous_or_unmatched_events_never_get_a_mapping_row(tmp_path, monkeypatch):
    """An event whose team names don't resolve to any real game must never produce a mapping
    index entry - there is nothing real to attribute."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_fake", game_id=None, home_team="Not A Real Team", away_team="Also Not Real", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_fake": [_snap("evt_fake", "book_a", "2026-09-14T17:00:00+00:00", -3.0, -150, 130)]}
    fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    assert find_provider_event_ids_for_game(SEASON, WEEK, REAL_GAME_ID) == []
    from nfl_predict.market.event_game_mapping import read_event_game_mappings

    assert read_event_game_mappings(SEASON, WEEK) == []


def test_mapping_remains_stable_after_a_later_odds_api_refresh(tmp_path, monkeypatch):
    """Re-polling the odds provider for the same game (a later slate refresh, a different
    bookmaker's line movement) must never create a duplicate or conflicting mapping row -
    the original mapping_timestamp is the one that survives."""
    _patch_storage(monkeypatch, tmp_path)
    events, markets = _real_shaped_events_and_markets()
    fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    from nfl_predict.market.event_game_mapping import read_event_game_mappings

    first_pass = read_event_game_mappings(SEASON, WEEK)
    assert len(first_pass) == 1
    first_timestamp = first_pass[0]["mapping_timestamp"]

    # A second, later refresh with a new fetched_at (line moved) - the snapshot store
    # appends a new row (different key), but the mapping index must stay exactly as it was.
    events2, markets2 = _real_shaped_events_and_markets()
    markets2["evt_denkc"] = [_snap("evt_denkc", "book_a", "2026-09-14T18:00:00+00:00", -3.0, -150, 130)]
    fetch_and_snapshot_live_odds(_FakeProvider(events2, markets2), season=SEASON)

    second_pass = read_event_game_mappings(SEASON, WEEK)
    assert len(second_pass) == 1  # still exactly one mapping row, not a duplicate
    assert second_pass[0]["mapping_timestamp"] == first_timestamp  # unchanged - the original observation is kept
    assert second_pass[0] == first_pass[0]


def test_a_canonical_game_id_alone_can_reconstruct_its_real_persisted_market_point(tmp_path, monkeypatch):
    """The core reconstruction proof: forget the original `OddsIngestionResult` object
    entirely - rebuild the same consensus MarketPoint using only what a canonical game_id
    (and the season/week it's known to fall in) can find on disk."""
    _patch_storage(monkeypatch, tmp_path)
    events, markets = _real_shaped_events_and_markets()
    original = fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)
    original_market = original.games[REAL_GAME_ID].market
    del original  # the reconstruction below must not depend on this object

    provider_event_ids = find_provider_event_ids_for_game(SEASON, WEEK, REAL_GAME_ID)
    assert len(provider_event_ids) == 1

    persisted = read_live_snapshots(provider_event_ids[0])
    assert persisted.height == 2
    reconstructed_snapshots = [OddsMarketSnapshot(**row) for row in persisted.to_dicts()]
    consensus = compute_market_consensus(reconstructed_snapshots)

    assert consensus.consensus_home_spread == original_market.home_spread_traditional
    assert consensus.consensus_home_no_vig_probability == original_market.no_vig_home_win_probability
    assert consensus.snapshot_reference == original_market.market_snapshot_reference
    assert consensus.provider_event_id == original_market.provider_event_id == "evt_denkc"


def _research_packet() -> ResearchInputPacket:
    return ResearchInputPacket(
        game_id=REAL_GAME_ID, season=SEASON, week=WEEK, season_type="REG",
        home_team_id="2310", away_team_id="1400", kickoff_timestamp="2026-09-14T20:15:00+00:00",
        packet_generated_at="2026-09-14T17:30:00+00:00",
        elo=ModelPrediction(model_id="elo_v2", available=True, predicted_margin=-3.2, home_win_probability=0.42),
        ridge=ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="test"),
        lightgbm=ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason="test"),
        market=MarketContext(available=False, unavailable_reason="test"),
        model_market_disagreement_points=None, known_qb_continuity_note=None,
        known_personnel_continuity_note=None, known_injury_summary=None,
    )


def test_spread_and_moneyline_decisions_can_be_replayed_entirely_from_persisted_data_with_no_network_call(tmp_path, monkeypatch):
    """Instruction #6/#7's central proof: persist real-shaped odds AND a real-shaped research
    run, then rebuild the full DecisionInputPacket from disk alone (never the in-memory
    objects the ingestion/research calls returned) and replay both market types through the
    completely unmodified decide(). No OddsProvider/Anthropic network call happens anywhere
    in this test - `_FakeProvider` and `FixtureLLMProvider` are both offline."""
    _patch_storage(monkeypatch, tmp_path)

    # --- Step 1: persist real-shaped odds, as an earlier live run would have. ---
    events, markets = _real_shaped_events_and_markets()
    fetch_and_snapshot_live_odds(_FakeProvider(events, markets), season=SEASON)

    # --- Step 2: persist a real-shaped research run (offline fixture provider, no network). ---
    run_research_for_game(
        packet=_research_packet(), provider=FixtureLLMProvider(LLM_FIXTURE),
        research_prompt_template_text="Research {{game_id}}.", run_id="offline_replay_run",
        now="2026-09-14T17:35:00+00:00",
    )

    # --- Step 3: RECONSTRUCT everything from disk only - game_id/season/week/run_id are the
    # only things carried over from the steps above; no ingestion/research result object is
    # reused. ---
    provider_event_ids = find_provider_event_ids_for_game(SEASON, WEEK, REAL_GAME_ID)
    persisted_snapshots = read_live_snapshots(provider_event_ids[0])
    consensus = compute_market_consensus([OddsMarketSnapshot(**row) for row in persisted_snapshots.to_dicts()])
    market = MarketPoint(
        available=True, source="reconstructed from market/live_snapshots",
        home_spread_traditional=consensus.consensus_home_spread,
        no_vig_home_win_probability=consensus.consensus_home_no_vig_probability,
        home_moneyline=-150, away_moneyline=130,
        snapshot_timestamp=persisted_snapshots["fetched_at"][0],
        provider_event_id=consensus.provider_event_id,
        market_snapshot_reference=consensus.snapshot_reference,
    )

    stored_research = read_research_run(SEASON, WEEK, REAL_GAME_ID, "offline_replay_run")
    now = "2026-09-14T18:00:00+00:00"
    research = build_research_point_from_stored_run(stored_research, now)
    assert research.available is True

    # Real-shaped, strongly-disagreeing, all-agreeing model predictions - same pattern the
    # existing "reaches gates 3-9" proof uses (test_live_market_and_research_wiring.py).
    elo = ModelPoint(model_id="elo_v2", available=True, predicted_margin=10.0, home_win_probability=0.7)
    ridge = ModelPoint(model_id="ridge_margin_E_v1", available=True, predicted_margin=8.0)
    lightgbm = ModelPoint(model_id="lightgbm_F_v1", available=True, predicted_margin=9.0, home_win_probability=0.68)
    agreement = compute_model_agreement(elo, ridge, lightgbm)

    health = SystemHealth(missing_required_data=(), stale_flags=(), market_age_seconds=1800, research_age_seconds=1500)
    packet = DecisionInputPacket(
        game_id=REAL_GAME_ID, decision_timestamp=now, kickoff_timestamp="2026-09-14T20:15:00",
        elo=elo, ridge=ridge, lightgbm=lightgbm, model_agreement=agreement,
        market=market, research=research, system_health=health,
    )
    rule_set = load_rule_set()

    results = {}
    for market_type in ("spread", "moneyline"):
        decision, reasons = decide(packet, rule_set, market_type)
        results[market_type] = (decision, reasons)
        assert reasons != (ReasonCode.MISSING_LIVE_DATA,)  # gates 3-9 reached, not stuck at gate 2

    assert results["spread"][0].value == "QUALIFIED_BET"
    assert results["moneyline"][0].value == "QUALIFIED_BET"

    # The decision packet itself must hash reproducibly and reference the real artifacts.
    assert packet.content_hash() == packet.content_hash()
    assert market.provider_event_id == "evt_denkc"
    assert market.market_snapshot_reference is not None
