import os
import tempfile

import pytest

from nfl_predict.config import get_settings

_LIVE_CREDENTIAL_ENV_VARS = (
    "NFL_ODDS_API_KEY", "NFL_RESEARCH_LLM_API_KEY", "NFL_WEATHER_API_KEY",
    "NFL_NEWS_API_KEY", "NFL_INJURY_API_KEY",
)

_embedded_postgres = None  # module-level handle so pytest_unconfigure can stop it


def pytest_configure(config):
    """Phase 10: tests/storage/test_postgres_blob_store.py and
    tests/data/test_postgres_connection.py exercise the real Postgres backend and are
    skipped unless NFL_TEST_POSTGRES_URL is set. Runs before collection, so the skipif marks
    (evaluated at collection time) see the env var this sets. If the `postgres` extra isn't
    installed (pgserver missing) or NFL_TEST_POSTGRES_URL is already set (e.g. CI pointing at
    a real hosted instance), this is a silent no-op and those tests are skipped as normal -
    never a hard dependency of the base test suite."""
    global _embedded_postgres
    if os.environ.get("NFL_TEST_POSTGRES_URL"):
        return
    try:
        import pgserver
    except ImportError:
        return
    data_dir = tempfile.mkdtemp(prefix="nfl_predict_test_pg_")
    _embedded_postgres = pgserver.get_server(data_dir)
    os.environ["NFL_TEST_POSTGRES_URL"] = _embedded_postgres.get_uri()


def pytest_unconfigure(config):
    global _embedded_postgres
    if _embedded_postgres is not None:
        _embedded_postgres.cleanup()
        _embedded_postgres = None


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """get_settings() is process-cached; tests that patch env vars must not leak state."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_live_credentials(monkeypatch):
    """A real production incident (Phase 8A correction) caught the test suite making an
    ACTUAL, BILLED Anthropic call: a test built for an environment where no live credential
    was ever configured started silently making live calls once this session's `.env`/OS
    environment gained real, working keys - `get_production_odds_provider()`/
    `get_production_research_provider()` read straight from `get_settings()`, so ANY test
    that exercises them without explicit mocking inherits whatever the ambient environment
    happens to have.

    This autouse fixture removes that whole class of risk: every test starts with these
    vars deliberately unset, regardless of what's in `.env` or the OS environment, so
    `get_production_*_provider()` always resolves to its explicit UNAVAILABLE status unless
    a test deliberately opts back in (e.g. `monkeypatch.setenv(...)` after this fixture has
    already run, still combined with mocking the actual network call - see
    `tests/live/test_secrets_never_persisted.py` for the pattern). Tests that construct a
    provider directly with an explicit `api_key="..."` argument are unaffected either way,
    since that bypasses `get_settings()` entirely."""
    for var in _LIVE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Points NFL_DATA_DIR at a throwaway directory so data tests never touch the real
    data/ dir or leave state behind. get_settings() is re-read fresh because the autouse
    _reset_settings_cache fixture already cleared its cache for this test."""
    monkeypatch.setenv("NFL_DATA_DIR", str(tmp_path))
    return tmp_path
