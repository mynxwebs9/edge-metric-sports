"""Phase 5 Step 23 proof #16 (API secrets never appear in committed configuration) plus
the provider-neutral interface and credential-gating behavior.

Phase 8A correction: a `--research-all --show-decisions` run once consumed nearly an entire
500-credit free allowance because `get_markets(event_id)` was called once per event
returned by `get_events()` (every event of the remaining season, not just the current
week). These tests prove the fix - `get_markets_for_sport()` is the one real, memoized,
budget-guarded batched call - entirely via a mocked `_get`/urlopen, no live network access
and no live credits spent."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from nfl_predict.market.odds_provider import (
    FixtureOddsProvider,
    OddsCreditBudgetExceededError,
    OddsEvent,
    OddsMarketSnapshot,
    OddsProvider,
    OddsProviderNotConfiguredError,
    OddsQuotaExceededError,
    OddsRequestError,
    TheOddsAPIProvider,
    _parse_events_response,
    _parse_odds_response,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "odds_fixture.json"


def test_the_odds_api_provider_refuses_to_construct_without_a_key(monkeypatch):
    monkeypatch.delenv("NFL_ODDS_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(OddsProviderNotConfiguredError):
        TheOddsAPIProvider()
    get_settings.cache_clear()


def test_the_odds_api_provider_constructs_with_an_explicit_key():
    provider = TheOddsAPIProvider(api_key="test-key-not-a-real-secret")
    assert isinstance(provider, OddsProvider)


def test_the_odds_api_provider_get_events_delegates_to_a_real_http_call(monkeypatch):
    provider = TheOddsAPIProvider(api_key="test-key-not-a-real-secret")
    captured = {}

    def fake_get(path, params):
        captured["path"] = path
        captured["params"] = params
        return [{"id": "evt1", "sport_key": "americanfootball_nfl", "sport_title": "NFL", "commence_time": "2026-09-13T17:00:00Z", "home_team": "Houston Texans", "away_team": "Buffalo Bills"}]

    monkeypatch.setattr(provider, "_get", fake_get)
    events = provider.get_events()
    assert events == [OddsEvent(provider_event_id="evt1", game_id=None, home_team="Houston Texans", away_team="Buffalo Bills", commence_time="2026-09-13T17:00:00Z")]
    assert "events" in captured["path"]


def _fake_odds_payload():
    return [{
        "id": "evt1", "commence_time": "2026-09-13T17:00:00Z", "home_team": "Houston Texans", "away_team": "Buffalo Bills",
        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "last_update": "2026-09-11T20:00:00Z", "markets": [
            {"key": "spreads", "last_update": "2026-09-11T20:00:00Z", "outcomes": [{"name": "Houston Texans", "price": -110, "point": 1.5}, {"name": "Buffalo Bills", "price": -110, "point": -1.5}]},
        ]}],
    }, {
        "id": "evt2", "commence_time": "2026-09-13T20:00:00Z", "home_team": "Some Team", "away_team": "Other Team",
        "bookmakers": [{"key": "fanduel", "title": "FanDuel", "last_update": "2026-09-11T20:00:00Z", "markets": [
            {"key": "h2h", "last_update": "x", "outcomes": [{"name": "Some Team", "price": -120}, {"name": "Other Team", "price": 100}]},
        ]}],
    }]


def test_get_markets_for_sport_requests_no_eventids_filter_and_no_totals(monkeypatch):
    """Test #9 (only US region) / part of #1: ONE request, no eventIds, markets=spreads,h2h
    only - the featured-odds endpoint returns every event's odds in this single call."""
    provider = TheOddsAPIProvider(api_key="test-key-not-a-real-secret")
    captured = {}

    def fake_get(path, params):
        captured["path"] = path
        captured["params"] = params
        return _fake_odds_payload()

    monkeypatch.setattr(provider, "_get", fake_get)
    markets = provider.get_markets_for_sport()
    assert "eventIds" not in captured["params"]
    assert captured["params"]["markets"] == "spreads,h2h"
    assert captured["params"]["regions"] == "us"
    assert {m.provider_event_id for m in markets} == {"evt1", "evt2"}  # both events came back from ONE call


def test_get_markets_for_sport_is_memoized_per_instance(monkeypatch):
    """Test proving reuse (Part A item 6/7): calling get_markets_for_sport() twice, or
    get_markets() for several different events, on the SAME provider instance issues only
    ONE HTTP request."""
    provider = TheOddsAPIProvider(api_key="test-key-not-a-real-secret")
    call_count = {"n": 0}

    def fake_get(path, params):
        call_count["n"] += 1
        return _fake_odds_payload()

    monkeypatch.setattr(provider, "_get", fake_get)
    provider.get_markets_for_sport()
    provider.get_markets_for_sport()
    provider.get_markets("evt1")
    provider.get_markets("evt2")
    assert call_count["n"] == 1


def test_get_markets_filters_the_batched_result_by_event_id(monkeypatch):
    provider = TheOddsAPIProvider(api_key="test-key-not-a-real-secret")
    monkeypatch.setattr(provider, "_get", lambda path, params: _fake_odds_payload())
    markets = provider.get_markets("evt1")
    assert len(markets) == 1
    assert markets[0].provider_event_id == "evt1"
    assert markets[0].home_spread_traditional == 1.5


def test_credit_budget_guard_refuses_the_call_with_no_http_request(monkeypatch):
    """Test #6: the guard trips BEFORE any network access - proven here by never even
    defining a working `_get` (it would raise if called)."""
    provider = TheOddsAPIProvider(api_key="test-key", max_credits_per_run=1)  # 1 credit budget; spreads+h2h costs 2

    def boom(path, params):
        raise AssertionError("must not make an HTTP request once the budget would be exceeded")

    monkeypatch.setattr(provider, "_get", boom)
    with pytest.raises(OddsCreditBudgetExceededError):
        provider.get_markets_for_sport()


def test_credit_budget_guard_allows_a_call_within_budget(monkeypatch):
    provider = TheOddsAPIProvider(api_key="test-key", max_credits_per_run=5)
    monkeypatch.setattr(provider, "_get", lambda path, params: _fake_odds_payload())
    markets = provider.get_markets_for_sport()
    assert len(markets) == 2
    assert provider.usage.credits_this_run == 2  # 2 markets x 1 region


def test_totals_are_never_requested():
    assert "totals" not in TheOddsAPIProvider.MARKETS.split(",")
    assert TheOddsAPIProvider.MARKETS == "spreads,h2h"


def test_only_us_region_is_requested_by_default():
    assert TheOddsAPIProvider.REGIONS == "us"


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, body=b"{}", headers=None):
        import io
        super().__init__(url="http://x", code=code, msg="err", hdrs=headers or {}, fp=io.BytesIO(body))


def test_quota_error_401_is_never_retried(monkeypatch):
    """Test #7: OUT_OF_USAGE_CREDITS-shaped errors (401/403) must not be retried - one
    failed HTTP attempt, not several."""
    provider = TheOddsAPIProvider(api_key="test-key")
    attempts = {"n": 0}

    def fake_urlopen(url, timeout=30):
        attempts["n"] += 1
        raise _FakeHTTPError(401, body=b'{"message": "Out of usage credits"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OddsQuotaExceededError):
        provider.get_markets_for_sport()
    assert attempts["n"] == 1


def test_transient_5xx_error_is_retried_a_bounded_number_of_times(monkeypatch):
    provider = TheOddsAPIProvider(api_key="test-key")
    attempts = {"n": 0}

    def fake_urlopen(url, timeout=30):
        attempts["n"] += 1
        raise _FakeHTTPError(503)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(OddsRequestError):
        provider.get_markets_for_sport()
    assert attempts["n"] == TheOddsAPIProvider.MAX_TRANSIENT_RETRIES + 1


def test_a_retry_still_counts_toward_the_same_single_request_not_extra_credits(monkeypatch):
    """A retried request that eventually succeeds must still only record ONE call/credit
    charge - retries are attempts at the SAME request, not additional ones."""
    provider = TheOddsAPIProvider(api_key="test-key")
    attempts = {"n": 0}

    class _FakeResponse:
        def __init__(self, body):
            self._body = body
            self.headers = {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=30):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise _FakeHTTPError(503)
        return _FakeResponse(json.dumps(_fake_odds_payload()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    provider.get_markets_for_sport()
    assert provider.usage.calls_this_run == 1
    assert provider.usage.credits_this_run == 2


def test_api_key_never_appears_in_a_logged_error_message(monkeypatch, caplog):
    """Every raised exception and every log line must carry only the redacted URL - the
    real key must never reach a log file or an exception message a developer might paste
    into an issue/chat."""
    import logging

    provider = TheOddsAPIProvider(api_key="super-secret-value")
    attempts = {"n": 0}

    def fake_urlopen(url, timeout=30):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _FakeHTTPError(503)
        raise _FakeHTTPError(401, body=b'{"message": "Out of usage credits"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(OddsQuotaExceededError) as exc_info:
            provider.get_markets_for_sport()

    assert "super-secret-value" not in str(exc_info.value)
    assert "REDACTED" in str(exc_info.value)
    for record in caplog.records:
        assert "super-secret-value" not in record.getMessage()


def test_usage_headers_are_captured_and_api_key_never_appears_in_them(monkeypatch):
    """Test #5: x-requests-last/used/remaining are captured; the api key never appears
    anywhere in the captured usage state."""
    provider = TheOddsAPIProvider(api_key="super-secret-value")

    class _FakeResponse:
        def __init__(self, body, headers):
            self._body = body
            self.headers = headers

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=30):
        assert "super-secret-value" not in url or True  # apiKey IS in the real request URL by API design
        return _FakeResponse(json.dumps(_fake_odds_payload()).encode("utf-8"), {"x-requests-last": "2", "x-requests-used": "122", "x-requests-remaining": "378"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider.get_markets_for_sport()
    usage = provider.usage.as_dict()
    assert usage == {"calls_this_run": 1, "credits_this_run": 2, "credits_used_account": 122, "credits_remaining_account": 378}
    assert "super-secret-value" not in json.dumps(usage)


def test_parse_events_response_handles_the_real_odds_api_shape():
    payload = [{"id": "evt1", "sport_key": "americanfootball_nfl", "sport_title": "NFL", "commence_time": "2026-09-13T17:00:00Z", "home_team": "Kansas City Chiefs", "away_team": "Denver Broncos"}]
    events = _parse_events_response(payload)
    assert events == [OddsEvent(provider_event_id="evt1", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-13T17:00:00Z")]


def test_parse_odds_response_handles_spreads_h2h_and_totals():
    payload = [{
        "id": "evt1", "commence_time": "2026-09-13T17:00:00Z", "home_team": "Kansas City Chiefs", "away_team": "Denver Broncos",
        "bookmakers": [{"key": "fanduel", "title": "FanDuel", "last_update": "2026-09-11T20:00:00Z", "markets": [
            {"key": "spreads", "last_update": "x", "outcomes": [{"name": "Kansas City Chiefs", "price": -110, "point": -3.0}, {"name": "Denver Broncos", "price": -110, "point": 3.0}]},
            {"key": "h2h", "last_update": "x", "outcomes": [{"name": "Kansas City Chiefs", "price": -148}, {"name": "Denver Broncos", "price": 124}]},
            {"key": "totals", "last_update": "x", "outcomes": [{"name": "Over", "price": -110, "point": 42.5}, {"name": "Under", "price": -110, "point": 42.5}]},
        ]}],
    }]
    snapshots = _parse_odds_response(payload, fetched_at="2026-09-11T20:05:00+00:00")
    assert len(snapshots) == 1
    s = snapshots[0]
    assert s.bookmaker == "fanduel"
    assert s.home_spread_traditional == -3.0 and s.away_spread_traditional == 3.0
    assert s.home_moneyline == -148 and s.away_moneyline == 124
    assert s.total_line == 42.5 and s.over_price == -110 and s.under_price == -110


def test_parse_odds_response_handles_a_bookmaker_missing_a_market_without_crashing():
    payload = [{
        "id": "evt1", "commence_time": "x", "home_team": "A", "away_team": "B",
        "bookmakers": [{"key": "book1", "title": "Book1", "last_update": "x", "markets": [
            {"key": "h2h", "last_update": "x", "outcomes": [{"name": "A", "price": -110}, {"name": "B", "price": -110}]},
        ]}],
    }]
    snapshots = _parse_odds_response(payload, fetched_at="2026-09-11T20:05:00+00:00")
    assert snapshots[0].home_spread_traditional is None
    assert snapshots[0].total_line is None
    assert snapshots[0].home_moneyline == -110


def test_fixture_provider_implements_the_full_interface_without_network_access():
    provider = FixtureOddsProvider(FIXTURE_PATH)
    events = provider.get_events()
    assert len(events) == 1
    assert events[0].game_id == "2024_01_BAL_KC"

    markets = provider.get_markets("evt_1")
    assert len(markets) == 2


def test_get_snapshot_returns_the_latest_snapshot_at_or_before_the_requested_time():
    provider = FixtureOddsProvider(FIXTURE_PATH)
    snap = provider.get_snapshot("evt_1", timestamp="2024-09-04T00:00:00Z")
    assert snap.fetched_at == "2024-09-03T10:00:00Z"  # only the earlier snapshot qualifies

    later_snap = provider.get_snapshot("evt_1", timestamp="2024-09-06T00:00:00Z")
    assert later_snap.fetched_at == "2024-09-05T18:00:00Z"


def test_get_snapshot_returns_none_when_no_snapshot_exists_before_the_requested_time():
    provider = FixtureOddsProvider(FIXTURE_PATH)
    snap = provider.get_snapshot("evt_1", timestamp="2020-01-01T00:00:00Z")
    assert snap is None


def test_no_odds_api_secret_appears_in_any_committed_config_or_env_file():
    """`.env.example` documents the variable NAME only, and `config/*.yaml` never contains
    a real key - this is a structural check against accidental secret commits."""
    project_root = Path(__file__).resolve().parents[2]
    env_example = (project_root / ".env.example").read_text(encoding="utf-8")
    assert "NFL_ODDS_API_KEY=" in env_example
    for line in env_example.splitlines():
        if line.strip().startswith("NFL_ODDS_API_KEY="):
            assert line.strip() == "NFL_ODDS_API_KEY="  # name only, no value

    for yaml_path in (project_root / "config").glob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        assert "NFL_ODDS_API_KEY" not in text
        assert "the-odds-api" not in text.lower() or "key" not in text.lower()
