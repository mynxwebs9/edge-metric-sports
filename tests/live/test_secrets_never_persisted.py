"""Phase 8A Step 28 proof #10: API secrets are never persisted in any live pipeline
artifact (automation report, health status, decision log).

Deliberately sets a FAKE (but present) key so `get_production_odds_provider()`/
`get_production_research_provider()` resolve to AVAILABLE, exercising the real
construction path - but `urllib.request.urlopen` is mocked so this never becomes an actual
network call (a real production incident, Phase 8A correction, was exactly a test making a
genuine live call once real credentials were configured in the ambient environment; this
test's own fake key must never reach the real API either)."""

from __future__ import annotations

import io
import json
import urllib.error

from nfl_predict.live import run as run_module

FAKE_SECRET = "sk-fake-secret-value-should-never-appear-anywhere"


def _fake_urlopen(*args, **kwargs):
    raise urllib.error.HTTPError(url="http://x", code=401, msg="Unauthorized", hdrs={}, fp=io.BytesIO(b"{}"))


def test_automation_report_never_contains_the_configured_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_ODDS_API_KEY", FAKE_SECRET)
    monkeypatch.setenv("NFL_RESEARCH_LLM_API_KEY", FAKE_SECRET)
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)  # never a real network call, even with this fake key
    # The fake key still makes it past provider construction, so a real (mocked-401) research
    # attempt runs far enough to persist a FailedResearchRun - without this, it silently wrote
    # into this repo's actual data/research/ directory every time the suite ran (a real,
    # confirmed leak found while building the Phase 8B API against real persisted data).
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    result = run_module.run_slate(game_id="2026_01_DEN_KC", dry_run=True)
    serialized = json.dumps(result, default=str)
    assert FAKE_SECRET not in serialized
