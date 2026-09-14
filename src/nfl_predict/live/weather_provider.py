"""Phase 8A Step 12: live weather for outdoor games via Open-Meteo - free, no API key
required, so this is a genuinely live-testable provider (unlike the odds/LLM providers,
which are gated on credentials this environment doesn't have).

Extends `nfl_predict.research.current_data_providers.WeatherProvider`/`WeatherForecast`
(Phase 6) rather than inventing a parallel type. Dome games are handled explicitly - no
forecast is fetched for a permanently-closed roof, and the result says so rather than
fabricating a number. Weather is structured context/research input only; nothing here
feeds a numeric model adjustment (no quantitative model currently accepts a weather input).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from nfl_predict.config import get_venues_config
from nfl_predict.research.current_data_providers import WeatherForecast, WeatherProvider

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,windspeed_10m,windgusts_10m,precipitation_probability"


class VenueNotFoundError(Exception):
    """Raised when a team_id has no entry in config/venues.yaml."""


def get_venue(team_id: str) -> dict:
    venues = get_venues_config()["venues"]
    if team_id not in venues:
        raise VenueNotFoundError(f"No venue configured for team_id={team_id!r} in config/venues.yaml")
    return venues[team_id]


def _nearest_hour_index(hourly_times: list[str], kickoff_iso: str) -> int:
    kickoff_dt = datetime.fromisoformat(kickoff_iso)
    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    diffs = []
    for t in hourly_times:
        t_dt = datetime.fromisoformat(t)
        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(tzinfo=timezone.utc)
        diffs.append(abs((t_dt - kickoff_dt).total_seconds()))
    return diffs.index(min(diffs))


class OpenMeteoWeatherProvider(WeatherProvider):
    """No credentials required - Open-Meteo's free tier needs no API key."""

    def get_forecast_for_venue(self, home_team_id: str, venue_name: str, kickoff_timestamp: str) -> WeatherForecast | None:
        venue = get_venue(home_team_id)
        retrieved_at = datetime.now(timezone.utc).isoformat()

        if venue["roof"] == "dome":
            return WeatherForecast(
                venue=venue_name, forecast_timestamp=retrieved_at, kickoff_timestamp=kickoff_timestamp,
                temperature_f=None, wind_mph=None, wind_gust_mph=None, precipitation_probability=None, roof_status="dome",
            )

        query = urllib.parse.urlencode({
            "latitude": venue["lat"], "longitude": venue["lon"], "hourly": HOURLY_FIELDS,
            "temperature_unit": "fahrenheit", "windspeed_unit": "mph", "timezone": "UTC", "forecast_days": 16,
        })
        url = f"{OPEN_METEO_BASE_URL}?{query}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None
        idx = _nearest_hour_index(times, kickoff_timestamp)

        return WeatherForecast(
            venue=venue_name, forecast_timestamp=retrieved_at, kickoff_timestamp=kickoff_timestamp,
            temperature_f=hourly.get("temperature_2m", [None] * len(times))[idx],
            wind_mph=hourly.get("windspeed_10m", [None] * len(times))[idx],
            wind_gust_mph=hourly.get("windgusts_10m", [None] * len(times))[idx],
            precipitation_probability=hourly.get("precipitation_probability", [None] * len(times))[idx],
            roof_status=venue["roof"],
        )

    def get_forecast(self, game_id: str) -> WeatherForecast | None:
        raise NotImplementedError("Use get_forecast_for_venue directly, or nfl_predict.live.run's orchestration, which resolves game_id -> (home_team_id, venue, kickoff) via the schedule before calling it.")
