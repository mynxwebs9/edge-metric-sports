from pathlib import Path

import pytest

from nfl_predict.config import (
    get_decision_thresholds_config,
    get_features_config,
    get_settings,
    get_sources_config,
    load_yaml_config,
)


def test_load_yaml_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("this_config_does_not_exist")


def test_features_config_schema():
    config = get_features_config()
    assert set(config.keys()) == {"features"}
    assert len(config["features"]) > 0
    names = [f["name"] for f in config["features"]]
    assert "off_epa_pp_season" in names  # spot check a real Phase 2 feature name


def test_sources_config_schema():
    config = get_sources_config()
    assert set(config.keys()) == {"sources"}
    names = [source["name"] for source in config["sources"]]
    assert "nflverse" in names


def test_nflverse_source_entry_has_all_required_fields():
    config = get_sources_config()
    nflverse = next(s for s in config["sources"] if s["name"] == "nflverse")
    required_fields = {
        "name", "description", "access_method", "auth_required", "auth_env_var",
        "update_frequency", "historical_coverage", "trust_level", "used_for",
        "known_issues", "raw_storage_convention",
    }
    assert required_fields <= set(nflverse.keys())
    assert nflverse["auth_required"] is False


def test_decision_thresholds_config_has_all_markets_unset():
    config = get_decision_thresholds_config()
    assert set(config["markets"]) == {"moneyline", "spread", "total"}
    for market_config in config["markets"].values():
        assert market_config["bet_min_edge"] is None
        assert market_config["lean_min_edge"] is None
        assert market_config["veto_conditions"] == []


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("NFL_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("NFL_DATABASE_URL", raising=False)
    monkeypatch.delenv("NFL_DATA_DIR", raising=False)
    monkeypatch.delenv("NFL_LOG_LEVEL", raising=False)

    settings = get_settings()

    assert settings.storage_backend == "sqlite"
    assert settings.database_url is None
    assert settings.log_level == "INFO"
    assert isinstance(settings.data_dir, Path)


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("NFL_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("NFL_DATABASE_URL", "postgresql://example/invalid")
    monkeypatch.setenv("NFL_LOG_LEVEL", "debug")

    settings = get_settings()

    assert settings.storage_backend == "postgres"
    assert settings.database_url == "postgresql://example/invalid"
    assert settings.log_level == "DEBUG"


def test_settings_rejects_invalid_storage_backend(monkeypatch):
    monkeypatch.setenv("NFL_STORAGE_BACKEND", "mongodb")

    with pytest.raises(ValueError, match="NFL_STORAGE_BACKEND"):
        get_settings()
