"""The Phase 4 unsealing gate: nothing may touch 2024-2025 before the freeze manifest
exists, is unmodified since it was frozen, and names every required candidate model."""

from __future__ import annotations

import json

import pytest

from nfl_predict.backtesting import holdout_guard
from nfl_predict.backtesting.freeze import (
    build_freeze_manifest,
    write_freeze_manifest,
)
from nfl_predict.models.elo import EloConfig

_ELO_CONFIG = EloConfig(k_factor=40.0, home_field_advantage=45.0)


def _valid_manifest() -> dict:
    return build_freeze_manifest(_ELO_CONFIG, {"method": "test"}, {"method": "test"})


def _point_guard_at(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    hash_path = tmp_path / "manifest.sha256"
    monkeypatch.setattr(holdout_guard, "freeze_manifest_paths", lambda: (manifest_path, hash_path))
    monkeypatch.setattr("nfl_predict.backtesting.freeze.freeze_manifest_paths", lambda: (manifest_path, hash_path))
    return manifest_path, hash_path


def test_assert_freeze_complete_raises_when_manifest_does_not_exist(monkeypatch, tmp_path):
    _point_guard_at(monkeypatch, tmp_path)
    with pytest.raises(holdout_guard.FreezeNotCompleteError, match="not found"):
        holdout_guard.assert_freeze_complete()


def test_assert_freeze_complete_raises_when_hash_sidecar_missing(monkeypatch, tmp_path):
    manifest_path, _hash_path = _point_guard_at(monkeypatch, tmp_path)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    with pytest.raises(holdout_guard.FreezeNotCompleteError, match="not found"):
        holdout_guard.assert_freeze_complete()


def test_assert_freeze_complete_raises_on_tampered_manifest(monkeypatch, tmp_path):
    _point_guard_at(monkeypatch, tmp_path)
    result = write_freeze_manifest(_valid_manifest())

    tampered = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    tampered["candidate_models"][0]["hyperparameters"] = {"snuck_in": True}
    result.manifest_path.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(holdout_guard.FreezeNotCompleteError, match="does not match its recorded hash"):
        holdout_guard.assert_freeze_complete()


def test_assert_freeze_complete_raises_when_a_required_candidate_is_missing(monkeypatch, tmp_path):
    _point_guard_at(monkeypatch, tmp_path)
    manifest = _valid_manifest()
    manifest["candidate_models"] = [c for c in manifest["candidate_models"] if c["model_id"] != "lightgbm_F_v1"]
    write_freeze_manifest(manifest)

    with pytest.raises(holdout_guard.FreezeNotCompleteError, match="lightgbm_F_v1"):
        holdout_guard.assert_freeze_complete()


def test_assert_freeze_complete_succeeds_on_a_freshly_written_valid_manifest(monkeypatch, tmp_path):
    _point_guard_at(monkeypatch, tmp_path)
    write_freeze_manifest(_valid_manifest())
    result = holdout_guard.assert_freeze_complete()
    assert isinstance(result, dict)
    assert len(result["candidate_models"]) == len(_valid_manifest()["candidate_models"])


def test_unseal_and_load_functions_call_the_freeze_check_first(monkeypatch, tmp_path):
    """Proof that the sealed-holdout-touching entry points cannot be reached without
    going through assert_freeze_complete - patch it to raise unconditionally and confirm
    both wrapper functions propagate that failure instead of doing any real work."""
    def _always_fail():
        raise holdout_guard.FreezeNotCompleteError("blocked for this test")

    monkeypatch.setattr(holdout_guard, "assert_freeze_complete", _always_fail)

    with pytest.raises(holdout_guard.FreezeNotCompleteError):
        holdout_guard.unseal_and_write_holdout_targets()

    with pytest.raises(holdout_guard.FreezeNotCompleteError):
        holdout_guard.load_holdout_game_dataset([2024])


def test_the_real_project_freeze_manifest_currently_verifies():
    """Integration check against the actual persisted Phase 4 artifact (not a mock) -
    if this fails, the real freeze is broken and 2024-2025 must not be unsealed."""
    manifest = holdout_guard.assert_freeze_complete()
    assert manifest["sealed_seasons_at_freeze_time"] == [2024, 2025]
