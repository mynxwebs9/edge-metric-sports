"""Phase 8A correction Step 1-2, tests #2-3, #8: real per-game odds -> real MarketPoints,
persisted through the existing append-only snapshot store first; a game with no usable
market stays unavailable rather than being fabricated.

Phase 8A correction (credit-exhaustion postmortem, see
docs/PHASE8A_LIVE_PIPELINE_REPORT.md): also proves `fetch_and_snapshot_live_odds` calls
`get_markets_for_sport()` (the real, credit-costing batched call) EXACTLY ONCE per
invocation, regardless of how many events/games are involved - never once per event."""

from __future__ import annotations

from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider

REAL_GAME_ID = "2026_01_DEN_KC"  # Kansas City Chiefs (home) vs Denver Broncos (away), season 2026 - verified against the real teams/games tables.


def _patch_storage(monkeypatch, tmp_path):
    """`fetch_and_snapshot_live_odds` now persists to two separate stores (the raw
    provider-native snapshots, and the additive event/game mapping index added by the
    Phase 8A provenance correction) - both must be redirected to `tmp_path`, or a test run
    silently writes real-looking fixture rows into this repo's actual `data/` directory."""
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


class _FakeProvider(OddsProvider):
    """A minimal in-memory OddsProvider - no network, no fixture file - so these tests
    exercise `fetch_and_snapshot_live_odds`'s own logic without depending on the real
    TheOddsAPIProvider's HTTP call. `get_markets_for_sport_calls` counts real "HTTP" calls
    this fake would have made - the number that must stay at 1 regardless of how many
    events/games are processed."""

    def __init__(self, events: list[OddsEvent], markets_by_event: dict[str, list[OddsMarketSnapshot]]):
        self._events = events
        self._markets = markets_by_event
        self.get_markets_for_sport_calls = 0

    def get_events(self) -> list[OddsEvent]:
        return self._events

    def get_markets_for_sport(self) -> list[OddsMarketSnapshot]:
        self.get_markets_for_sport_calls += 1
        return [s for snaps in self._markets.values() for s in snaps]


def _snap(event_id, bookmaker, fetched_at="2026-09-12T10:00:00+00:00", home_spread=-3.0, home_ml=-150, away_ml=130) -> OddsMarketSnapshot:
    return OddsMarketSnapshot(
        provider_event_id=event_id, fetched_at=fetched_at, bookmaker=bookmaker,
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread if home_spread is not None else None,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=home_ml, away_moneyline=away_ml,
        total_line=44.5, over_price=-110, under_price=-110,
    )


def test_provider_available_alone_does_not_populate_any_market(tmp_path, monkeypatch):
    """Test #1: an OddsProvider with zero events (the provider 'works' - it responds - but
    has nothing for this slate) must never be interpreted as real market data existing."""
    _patch_storage(monkeypatch, tmp_path)
    provider = _FakeProvider(events=[], markets_by_event={})
    result = fetch_and_snapshot_live_odds(provider, season=2026)
    assert result.games == {}
    assert result.events_matched == 0


def test_real_valid_per_game_odds_make_marketpoint_available(tmp_path, monkeypatch):
    """Test #2."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_denkc": [_snap("evt_denkc", "book_a"), _snap("evt_denkc", "book_b", home_spread=-3.5, home_ml=-145, away_ml=125)]}
    provider = _FakeProvider(events, markets)

    result = fetch_and_snapshot_live_odds(provider, season=2026)

    assert REAL_GAME_ID in result.games
    market = result.games[REAL_GAME_ID].market
    assert market.available is True
    assert market.home_spread_traditional == -3.25  # median of [-3.0, -3.5]
    assert market.no_vig_home_win_probability is not None
    assert market.snapshot_timestamp == "2026-09-12T10:00:00+00:00"
    assert "market/live_snapshots/event=evt_denkc" in market.source
    assert result.games[REAL_GAME_ID].n_books == 2
    assert provider.get_markets_for_sport_calls == 1


def test_a_full_slate_of_many_events_still_uses_exactly_one_odds_fetch(tmp_path, monkeypatch):
    """Test proving the actual credit-exhaustion fix: a slate of many events (simulating an
    entire remaining season's worth, not just one week) still results in exactly ONE
    get_markets_for_sport() call, never one per event."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id=f"evt_{i}", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z") for i in range(150)]
    markets = {f"evt_{i}": [_snap(f"evt_{i}", "book_a")] for i in range(150)}
    provider = _FakeProvider(events, markets)

    fetch_and_snapshot_live_odds(provider, season=2026)

    assert provider.get_markets_for_sport_calls == 1


def test_game_ids_scopes_which_matched_events_are_processed(tmp_path, monkeypatch):
    """`game_ids` narrows which mapped games get persisted/returned - it does not change the
    number of HTTP calls (already fixed at 1), just what this run actually cares about."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_denkc": [_snap("evt_denkc", "book_a")]}
    provider = _FakeProvider(events, markets)

    result = fetch_and_snapshot_live_odds(provider, season=2026, game_ids=["some_other_game"])

    assert REAL_GAME_ID not in result.games  # not in the requested scope, even though it matched
    assert provider.get_markets_for_sport_calls == 1


def test_snapshots_are_persisted_through_the_append_only_store_before_use(tmp_path, monkeypatch):
    """Test #8: the market snapshot must exist in the append-only store before the decision
    packet can reference it."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_denkc": [_snap("evt_denkc", "book_a")]}
    provider = _FakeProvider(events, markets)

    result = fetch_and_snapshot_live_odds(provider, season=2026)

    from nfl_predict.market.live_snapshot_store import read_live_snapshots

    persisted = read_live_snapshots("evt_denkc")
    assert persisted.height == 1
    assert persisted["bookmaker"][0] == "book_a"
    assert len(result.games[REAL_GAME_ID].snapshot_paths) == 1


def test_a_game_with_no_usable_market_data_stays_unmatched_not_fabricated(tmp_path, monkeypatch):
    """Test #3 (partial - the missing-market side): a bookmaker entry with neither a spread
    nor a moneyline for this event must not produce a fake `available=True` MarketPoint."""
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    empty_snap = OddsMarketSnapshot(
        provider_event_id="evt_denkc", fetched_at="2026-09-12T10:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=None, away_spread_traditional=None, home_spread_price=None, away_spread_price=None,
        home_moneyline=None, away_moneyline=None, total_line=None, over_price=None, under_price=None,
    )
    provider = _FakeProvider(events, {"evt_denkc": [empty_snap]})

    result = fetch_and_snapshot_live_odds(provider, season=2026)

    assert REAL_GAME_ID not in result.games
    assert "evt_denkc" in result.events_without_usable_markets


def test_an_event_with_unmapped_team_names_is_skipped_not_guessed(tmp_path, monkeypatch):
    _patch_storage(monkeypatch, tmp_path)
    events = [OddsEvent(provider_event_id="evt_x", game_id=None, home_team="Not A Real Team", away_team="Also Not Real", commence_time="2026-09-14T20:15:00Z")]
    provider = _FakeProvider(events, {"evt_x": [_snap("evt_x", "book_a")]})

    result = fetch_and_snapshot_live_odds(provider, season=2026)

    assert result.games == {}
    assert "Also Not Real @ Not A Real Team" in result.events_unmatched


def test_research_packet_construction_never_imports_an_odds_provider():
    """Test #4 (research packets do not trigger new odds HTTP calls): a structural proof -
    research_live.py's own CODE (not its docstrings) never imports anything from
    market.odds_provider/live.odds_ingestion, so it is IMPOSSIBLE for building a research
    packet to trigger a fresh odds fetch; it can only consume the MarketPoint run.py already
    computed once and passed in."""
    import ast
    import inspect

    from nfl_predict.live import research_live

    tree = ast.parse(inspect.getsource(research_live))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)

    assert not any("odds_provider" in m or "odds_ingestion" in m for m in imported_modules)
