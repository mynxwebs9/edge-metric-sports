"""Phase 8A Step 28 proof #27: weather timestamps are preserved; dome games never get a
fabricated forecast."""

from __future__ import annotations

import json

import pytest

from nfl_predict.live.weather_provider import OpenMeteoWeatherProvider, VenueNotFoundError, get_venue


def test_get_venue_returns_real_configured_coordinates():
    venue = get_venue("2310")  # KC
    assert venue["stadium"] == "GEHA Field at Arrowhead Stadium"
    assert venue["roof"] == "outdoors"


def test_get_venue_raises_for_an_unknown_team_id():
    with pytest.raises(VenueNotFoundError):
        get_venue("nonexistent_team_id")


def test_dome_venue_returns_no_forecast_numbers_never_fabricated():
    provider = OpenMeteoWeatherProvider()
    forecast = provider.get_forecast_for_venue("2520", "Allegiant Stadium", "2026-09-14T20:15:00")
    assert forecast.roof_status == "dome"
    assert forecast.temperature_f is None
    assert forecast.wind_mph is None
    assert forecast.wind_gust_mph is None
    assert forecast.precipitation_probability is None


def test_outdoor_venue_preserves_both_forecast_and_kickoff_timestamps(monkeypatch):
    """Forecast (retrieval) timestamp and kickoff timestamp are distinct fields, both
    preserved - never collapsed into one or overwritten. Also proves the real kickoff-time
    bug fix: "2026-09-14T20:15:00" is the schedule's ambiguous Eastern-time kickoff (nflverse
    convention), whose REAL UTC instant is 2026-09-15T00:15:00 - Open-Meteo's hourly buckets
    are genuine UTC, so matching the nearest hour only works correctly once kickoff is
    resolved to UTC first (the bug this replaced compared them unresolved and picked a
    forecast hour 4-5 hours off)."""
    fake_payload = {
        "hourly": {
            "time": ["2026-09-14T23:00", "2026-09-15T00:00", "2026-09-15T01:00"],
            "temperature_2m": [70.0, 72.0, 74.0],
            "windspeed_10m": [5.0, 6.0, 7.0],
            "windgusts_10m": [10.0, 11.0, 12.0],
            "precipitation_probability": [0, 5, 10],
        }
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(fake_payload).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=30: FakeResponse())

    provider = OpenMeteoWeatherProvider()
    forecast = provider.get_forecast_for_venue("2310", "GEHA Field at Arrowhead Stadium", "2026-09-14T20:15:00")
    assert forecast.kickoff_timestamp == "2026-09-14T20:15:00"
    assert forecast.forecast_timestamp != forecast.kickoff_timestamp
    # real kickoff (00:15 UTC) is closest to the 00:00 UTC hourly bucket
    assert forecast.temperature_f == 72.0
    assert forecast.wind_gust_mph == 11.0


def test_get_forecast_by_game_id_is_not_implemented_directly():
    provider = OpenMeteoWeatherProvider()
    with pytest.raises(NotImplementedError):
        provider.get_forecast("2026_01_DEN_KC")
