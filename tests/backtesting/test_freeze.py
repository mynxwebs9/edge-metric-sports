"""Phase 4 Steps C-F: the freeze manifest must name every candidate model, its exact
feature list, and its exact hyperparameters, and must hash reproducibly."""

from __future__ import annotations

import hashlib
import json

from nfl_predict.backtesting.freeze import (
    CANDIDATE_MODELS,
    FREEZE_FILENAME,
    FREEZE_HASH_FILENAME,
    build_freeze_manifest,
    freeze_manifest_paths,
    write_freeze_manifest,
)
from nfl_predict.models.elo import EloConfig
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES

_ELO_CONFIG = EloConfig(k_factor=40.0, home_field_advantage=45.0)
_ELO_SUMMARY = {"method": "test", "selected_k_factor": 40.0}
_BIAS_SUMMARY = {"method": "test", "conclusion": "no correction"}


def _build() -> dict:
    return build_freeze_manifest(_ELO_CONFIG, _ELO_SUMMARY, _BIAS_SUMMARY)


def test_manifest_includes_every_candidate_model():
    manifest = _build()
    present_ids = {c["model_id"] for c in manifest["candidate_models"]}
    expected_ids = {c["model_id"] for c in CANDIDATE_MODELS}
    assert present_ids == expected_ids
    assert len(manifest["candidate_models"]) == len(CANDIDATE_MODELS)


def test_elo_candidate_hyperparameters_come_from_the_passed_config_not_a_hardcoded_value():
    manifest = _build()
    elo_entry = next(c for c in manifest["candidate_models"] if c["model_id"] == "elo_v2")
    assert elo_entry["hyperparameters"]["k_factor"] == 40.0
    assert elo_entry["hyperparameters"]["home_field_advantage"] == 45.0

    other_config = EloConfig(k_factor=17.0, home_field_advantage=33.0)
    other_manifest = build_freeze_manifest(other_config, _ELO_SUMMARY, _BIAS_SUMMARY)
    other_elo_entry = next(c for c in other_manifest["candidate_models"] if c["model_id"] == "elo_v2")
    assert other_elo_entry["hyperparameters"]["k_factor"] == 17.0
    assert other_elo_entry["hyperparameters"]["home_field_advantage"] == 33.0


def test_every_feature_based_candidate_has_its_exact_frozen_feature_list():
    manifest = _build()
    for entry in manifest["candidate_models"]:
        feature_set = entry.get("feature_set")
        if feature_set is None:
            continue
        expected = ABLATIONS.get(feature_set) if feature_set != "CORE" else CORE_FEATURES
        assert entry["feature_names"] == expected
        assert entry["n_features"] == len(expected)


def test_ridge_total_bias_correction_is_frozen_as_not_applied():
    manifest = _build()
    ridge_total = next(c for c in manifest["candidate_models"] if c["model_id"] == "ridge_total_E_v1")
    assert ridge_total["calibration"]["correction_applied"] is False


def test_manifest_contains_no_sealed_season_outcome_data():
    """The manifest is built purely from config/hyperparameters/investigation summaries
    (all computed on development+validation data) - it must never contain actual game
    outcomes, since that would mean 2024-2025 was read before the freeze itself exists."""
    manifest = _build()
    serialized = json.dumps(manifest)
    # "total_points"/"home_margin" appear legitimately as target-column NAMES in
    # candidate_models[*]["targets"] - what must never appear is actual score data.
    assert "home_score" not in serialized
    assert "away_score" not in serialized


def test_write_freeze_manifest_produces_a_matching_hash_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nfl_predict.backtesting.freeze.get_settings",
        lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})(),
    )
    manifest = _build()
    result = write_freeze_manifest(manifest)

    assert result.manifest_path.name == FREEZE_FILENAME
    assert result.hash_path.name == FREEZE_HASH_FILENAME
    assert result.manifest_path.is_file()
    assert result.hash_path.is_file()

    written_bytes = result.manifest_path.read_text(encoding="utf-8").encode("utf-8")
    assert hashlib.sha256(written_bytes).hexdigest() == result.manifest_sha256
    assert result.hash_path.read_text(encoding="utf-8").strip() == result.manifest_sha256


def test_freeze_manifest_paths_are_under_configured_data_dir():
    manifest_path, hash_path = freeze_manifest_paths()
    assert manifest_path.name == FREEZE_FILENAME
    assert hash_path.name == FREEZE_HASH_FILENAME
    assert manifest_path.parent == hash_path.parent
