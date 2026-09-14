"""Phase 5 Step 21: a provider-neutral live odds interface.

`OddsProvider` is the abstract contract every live-odds adapter must implement -
`get_events()`, `get_markets_for_sport()`, `get_markets(provider_event_id)`,
`get_snapshot(provider_event_id, timestamp)` - so nothing else in the platform hard-codes a
specific vendor. `TheOddsAPIProvider` is the first real adapter (chosen per
`docs/DATA_SOURCES.md`); `FixtureOddsProvider` is a second, offline adapter that reads canned
JSON fixtures and makes no network call - used to exercise the interface end-to-end without
live credentials.

Credentials come ONLY from `nfl_predict.config.get_settings().odds_api_key` (itself sourced
from the `NFL_ODDS_API_KEY` environment variable, per `.env.example`) - never hard-coded,
never read directly from `os.environ` in this module. If that key is unset,
`TheOddsAPIProvider` raises `OddsProviderNotConfiguredError` on construction rather than
fabricating a response or silently falling back to fixture data.

**Phase 8A correction - credit-exhaustion postmortem (see
docs/PHASE8A_LIVE_PIPELINE_REPORT.md for the full incident report):** a single
`--research-all --show-decisions` run once consumed nearly an entire 500-credit free
allowance. Root cause: `get_markets(provider_event_id)` was called once per event returned
by `get_events()` (which returns EVERY upcoming event for the whole remaining season, not
just the current week), and each such call is a SEPARATE billed request to
`/sports/{sport}/odds` costing `len(markets) x len(regions)` credits regardless of the
`eventIds` filter - so a ~170-event remaining season at 3 markets (`spreads,h2h,totals`) x 1
region cost roughly 500 credits in one run. The fix: `get_markets_for_sport()` is now the
real, credit-costing call - ONE request, with NO `eventIds` filter, that returns odds for
EVERY event in a single response for the same `markets x regions` cost as fetching just one
event. `get_markets(provider_event_id)` is now a free, local filter over an
instance-memoized `get_markets_for_sport()` result - it makes its own HTTP call only if
`get_markets_for_sport()` has never been called on this instance yet, and even then it's the
same single batched request, never a per-event one. `MARKETS` is `spreads,h2h` only -
totals remain deferred and were never worth the extra credit multiplier. A hard per-run
credit budget (`NFL_ODDS_MAX_CREDITS_PER_RUN`, default 5) is enforced BEFORE any request that
would exceed it - refused locally, with zero additional HTTP calls, not merely logged after
the fact.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nfl_predict.config import get_settings
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)


class OddsProviderNotConfiguredError(Exception):
    """Raised when a live provider is used without its required credentials configured."""


class OddsQuotaExceededError(Exception):
    """Raised when The Odds API itself reports a credential/quota error (HTTP 401/403) -
    NEVER retried, per the correction brief's explicit "no retry storm" requirement. The
    account may be genuinely out of credits, or the key may be invalid - either way,
    retrying would only make things worse."""


class OddsRequestError(Exception):
    """Raised when a request ultimately fails for a reason other than quota/credentials
    (e.g. a transient error that exhausted its small retry budget, or an unexpected HTTP
    status)."""


class OddsCreditBudgetExceededError(Exception):
    """Raised BEFORE any HTTP request is made, when that request's estimated cost would
    exceed this run's configured `NFL_ODDS_MAX_CREDITS_PER_RUN` budget. This is a local,
    zero-network-cost refusal - the whole point of the budget guard is to never even attempt
    the request once the budget is exhausted."""


@dataclass
class OddsUsage:
    """Real, observed Odds-API usage for one provider instance (i.e. one pipeline run).
    `last_request_credits`/`account_credits_used`/`account_credits_remaining` come directly
    from the API's own `x-requests-last`/`x-requests-used`/`x-requests-remaining` response
    headers - `None` until at least one real response has been received; never estimated or
    fabricated. `calls_this_run`/`credits_this_run` are OUR OWN local counters (calls
    actually issued, and our own markets x regions cost estimate for each) - the mechanism
    the budget guard checks against, independent of what the API's headers later confirm."""

    calls_this_run: int = 0
    credits_this_run: int = 0
    last_request_credits: int | None = None
    account_credits_used: int | None = None
    account_credits_remaining: int | None = None

    def as_dict(self) -> dict:
        return {
            "calls_this_run": self.calls_this_run,
            "credits_this_run": self.credits_this_run,
            "credits_used_account": self.account_credits_used,
            "credits_remaining_account": self.account_credits_remaining,
        }


@dataclass(frozen=True)
class OddsEvent:
    provider_event_id: str
    game_id: str | None  # mapped to this project's game_id once matched against schedules; None if unmatched
    home_team: str
    away_team: str
    commence_time: str


@dataclass(frozen=True)
class OddsMarketSnapshot:
    provider_event_id: str
    fetched_at: str
    bookmaker: str
    home_spread_traditional: float | None
    away_spread_traditional: float | None
    home_spread_price: int | None
    away_spread_price: int | None
    home_moneyline: int | None
    away_moneyline: int | None
    total_line: float | None
    over_price: int | None
    under_price: int | None


class OddsProvider(ABC):
    @abstractmethod
    def get_events(self) -> list[OddsEvent]: ...

    @abstractmethod
    def get_markets_for_sport(self) -> list[OddsMarketSnapshot]:
        """Fetch odds for EVERY event of this sport in ONE call. This is the real,
        credit-costing operation - a provider must never issue this once per event in a
        loop (see this module's docstring for the incident that motivated this contract)."""
        ...

    def get_markets(self, provider_event_id: str) -> list[OddsMarketSnapshot]:
        """Default implementation: filter `get_markets_for_sport()`'s result down to one
        event. Callers that need every event's markets should call
        `get_markets_for_sport()` directly and filter/group locally instead of calling this
        once per event - a provider whose `get_markets_for_sport()` memoizes per instance
        (as `TheOddsAPIProvider` does) makes repeated calls to this method free after the
        first, but a full-slate loop calling this method is still exactly the pattern that
        caused the credit-exhaustion incident and must not be reintroduced."""
        return [m for m in self.get_markets_for_sport() if m.provider_event_id == provider_event_id]

    def get_snapshot(self, provider_event_id: str, timestamp: str) -> OddsMarketSnapshot | None:
        """Default implementation: the most recent snapshot from `get_markets()` at or
        before `timestamp`. A provider with server-side historical-snapshot support (e.g. a
        paid odds-history endpoint) may override this for an exact-timestamp fetch instead
        of filtering client-side."""
        candidates = [m for m in self.get_markets(provider_event_id) if m.fetched_at <= timestamp]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.fetched_at)


def _parse_events_response(payload: list[dict]) -> list[OddsEvent]:
    """Pure parsing of The Odds API's `/v4/sports/{sport}/events` response shape - no
    network I/O, so this is directly unit-testable against a realistic recorded/mocked
    response body without a live key."""
    return [
        OddsEvent(provider_event_id=e["id"], game_id=None, home_team=e["home_team"], away_team=e["away_team"], commence_time=e["commence_time"])
        for e in payload
    ]


def _parse_odds_response(payload: list[dict], fetched_at: str) -> list[OddsMarketSnapshot]:
    """Pure parsing of The Odds API's `/v4/sports/{sport}/odds` response shape (one entry
    per event, each with a `bookmakers` list of `{key, markets: [{key: "spreads"|"h2h"|
    "totals", outcomes: [{name, price, point}]}]}`) into one `OddsMarketSnapshot` per
    (event, bookmaker). `game_id` mapping happens later against this project's schedule,
    not here - this function only knows what the provider told it."""
    snapshots = []
    for event in payload:
        provider_event_id = event["id"]
        home_team, away_team = event["home_team"], event["away_team"]
        for bookmaker in event.get("bookmakers", []):
            home_spread = away_spread = home_spread_price = away_spread_price = None
            home_ml = away_ml = None
            total_line = over_price = under_price = None
            for market in bookmaker.get("markets", []):
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                if market["key"] == "spreads":
                    if home_team in outcomes:
                        home_spread, home_spread_price = outcomes[home_team].get("point"), outcomes[home_team].get("price")
                    if away_team in outcomes:
                        away_spread, away_spread_price = outcomes[away_team].get("point"), outcomes[away_team].get("price")
                elif market["key"] == "h2h":
                    if home_team in outcomes:
                        home_ml = outcomes[home_team].get("price")
                    if away_team in outcomes:
                        away_ml = outcomes[away_team].get("price")
                elif market["key"] == "totals":
                    over = next((o for o in market.get("outcomes", []) if o["name"] == "Over"), None)
                    under = next((o for o in market.get("outcomes", []) if o["name"] == "Under"), None)
                    if over:
                        total_line, over_price = over.get("point"), over.get("price")
                    if under:
                        under_price = under.get("price")
            snapshots.append(OddsMarketSnapshot(
                provider_event_id=provider_event_id, fetched_at=fetched_at, bookmaker=bookmaker["key"],
                home_spread_traditional=home_spread, away_spread_traditional=away_spread,
                home_spread_price=home_spread_price, away_spread_price=away_spread_price,
                home_moneyline=home_ml, away_moneyline=away_ml,
                total_line=total_line, over_price=over_price, under_price=under_price,
            ))
    return snapshots


class TheOddsAPIProvider(OddsProvider):
    """The first real adapter - chosen per `docs/DATA_SOURCES.md`. Supports SPREAD and
    MONEYLINE (`markets=spreads,h2h` - totals remain deferred and are never requested, both
    because the decision engine doesn't support them yet and because every extra market
    multiplies credit cost). See this module's docstring for the credit-exhaustion incident
    that shaped this class's current design: ONE batched `get_markets_for_sport()` call,
    never one call per event."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "americanfootball_nfl"
    MARKETS = "spreads,h2h"
    REGIONS = "us"
    ODDS_FORMAT = "american"
    MAX_TRANSIENT_RETRIES = 2  # for 429/5xx only - never for 401/403 quota/credential errors

    def __init__(self, api_key: str | None = None, max_credits_per_run: int | None = None):
        self._api_key = api_key if api_key is not None else get_settings().odds_api_key
        if not self._api_key:
            raise OddsProviderNotConfiguredError(
                "The Odds API key is not configured - set NFL_ODDS_API_KEY in .env before "
                "using TheOddsAPIProvider. Live odds calls are refused, not fabricated."
            )
        self._max_credits_per_run = max_credits_per_run if max_credits_per_run is not None else get_settings().odds_max_credits_per_run
        self.usage = OddsUsage()
        self._markets_for_sport_cache: list[OddsMarketSnapshot] | None = None

    def _estimated_credits_for_odds_call(self) -> int:
        return len([m for m in self.MARKETS.split(",") if m]) * len([r for r in self.REGIONS.split(",") if r])

    def _record_usage_headers(self, headers) -> None:
        def _int_or_none(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        self.usage.last_request_credits = _int_or_none(headers.get("x-requests-last"))
        self.usage.account_credits_used = _int_or_none(headers.get("x-requests-used"))
        self.usage.account_credits_remaining = _int_or_none(headers.get("x-requests-remaining"))

    def _get(self, path: str, params: dict) -> object:
        # The API key is a query parameter for this provider (as documented by The Odds
        # API) - never logged, never written to any artifact; callers of this provider
        # only ever receive the parsed OddsEvent/OddsMarketSnapshot dataclasses, which have
        # no field for it (see tests/market/test_odds_provider.py's secret-artifact proof).
        # `redacted_url` (log/error-message use only) never contains the real key.
        query = {**params, "apiKey": self._api_key}
        url = f"{self.BASE_URL}{path}?{urllib.parse.urlencode(query)}"
        redacted_url = f"{self.BASE_URL}{path}?{urllib.parse.urlencode({**params, 'apiKey': 'REDACTED'})}"

        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # pragma: no cover - never exercised without a live key
                    body = json.loads(resp.read().decode("utf-8"))
                    self._record_usage_headers(resp.headers)
                    return body
            except urllib.error.HTTPError as e:  # pragma: no cover - never exercised without a live key
                self._record_usage_headers(e.headers)
                if e.code in (401, 403):
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                    raise OddsQuotaExceededError(
                        f"The Odds API returned HTTP {e.code} for {redacted_url} - treated as a "
                        f"credential/quota error and NEVER retried. Response body: {detail}"
                    ) from e
                if e.code == 429 or e.code >= 500:
                    if attempt > self.MAX_TRANSIENT_RETRIES:
                        raise OddsRequestError(
                            f"The Odds API returned HTTP {e.code} for {redacted_url} after "
                            f"{attempt} attempt(s) - transient-error retry budget exhausted."
                        ) from e
                    logger.warning("Odds API transient error %s on attempt %d/%d for %s - retrying", e.code, attempt, self.MAX_TRANSIENT_RETRIES + 1, redacted_url)
                    time.sleep(0.5 * attempt)
                    continue
                raise OddsRequestError(f"The Odds API returned unexpected HTTP {e.code} for {redacted_url}") from e

    def get_events(self) -> list[OddsEvent]:
        payload = self._get(f"/sports/{self.SPORT_KEY}/events", {})
        return _parse_events_response(payload)

    def get_markets_for_sport(self) -> list[OddsMarketSnapshot]:
        """The real, credit-costing call - ONE request, no `eventIds` filter, covering
        every event of the sport for `len(markets) x len(regions)` credits regardless of how
        many events come back. Memoized on this instance: calling this (or `get_markets()`)
        again on the SAME provider instance never issues a second HTTP request."""
        if self._markets_for_sport_cache is not None:
            return self._markets_for_sport_cache

        estimated_cost = self._estimated_credits_for_odds_call()
        if self.usage.credits_this_run + estimated_cost > self._max_credits_per_run:
            raise OddsCreditBudgetExceededError(
                f"Refusing to fetch odds: an estimated {estimated_cost} credit(s) would bring "
                f"this run's total to {self.usage.credits_this_run + estimated_cost}, exceeding "
                f"the configured NFL_ODDS_MAX_CREDITS_PER_RUN={self._max_credits_per_run}. "
                "No HTTP request was made."
            )

        payload = self._get(f"/sports/{self.SPORT_KEY}/odds", {"regions": self.REGIONS, "markets": self.MARKETS, "oddsFormat": self.ODDS_FORMAT})
        fetched_at = datetime.now(timezone.utc).isoformat()
        self.usage.calls_this_run += 1
        self.usage.credits_this_run += estimated_cost
        self._markets_for_sport_cache = _parse_odds_response(payload, fetched_at)
        return self._markets_for_sport_cache


class FixtureOddsProvider(OddsProvider):
    """Offline adapter for tests/local development - reads a local JSON fixture file and
    makes no network call. Exercises the `OddsProvider` interface end-to-end without live
    credentials. Fixture shape: `{"events": [...OddsEvent fields...], "markets": [...
    OddsMarketSnapshot fields...]}`."""

    def __init__(self, fixture_path: str | Path):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def get_events(self) -> list[OddsEvent]:
        return [OddsEvent(**e) for e in self._fixture.get("events", [])]

    def get_markets_for_sport(self) -> list[OddsMarketSnapshot]:
        return [OddsMarketSnapshot(**m) for m in self._fixture.get("markets", [])]
