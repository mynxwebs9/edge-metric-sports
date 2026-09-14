"""Phase 4 Step 13 proofs #6-7: predictions are persisted before outcomes are ever joined,
and scoring can never mutate a stored prediction."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from nfl_predict.backtesting.ledger import (
    LedgerImmutableError,
    ledger_paths,
    read_prediction_ledger,
    verify_ledger_integrity,
    write_prediction_ledger,
)
from nfl_predict.backtesting.walk_forward import PredictionRecord


def _sample_records(run_id_suffix: str) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            game_id=f"g{i}", season=2024, week=1, season_type="REG", kickoff_timestamp=None,
            model_id="test_model", model_version="v1", feature_version="v1", target="home_margin",
            predicted_value=float(i), training_cutoff_season=2023, training_cutoff_week=22,
            training_row_count=100, artifact_hash="deadbeef", feature_names_hash="feedface",
        )
        for i in range(3)
    ]


def test_write_prediction_ledger_refuses_to_overwrite_an_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.backtesting.ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    write_prediction_ledger(_sample_records("a"), run_id="run_x")
    with pytest.raises(LedgerImmutableError, match="already exists"):
        write_prediction_ledger(_sample_records("a"), run_id="run_x")


def test_read_prediction_ledger_never_writes_to_the_ledger_file(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.backtesting.ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    result = write_prediction_ledger(_sample_records("b"), run_id="run_y")
    before = result.ledger_path.read_bytes()

    for _ in range(3):
        read_prediction_ledger("run_y")

    after = result.ledger_path.read_bytes()
    assert before == after
    assert verify_ledger_integrity("run_y") is True


def test_verify_ledger_integrity_detects_any_post_write_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.backtesting.ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    result = write_prediction_ledger(_sample_records("c"), run_id="run_z")

    frame = pl.read_parquet(result.ledger_path, memory_map=False)
    mutated = frame.with_columns(pl.col("predicted_value") * 0 + 999.0)
    mutated.write_parquet(result.ledger_path)  # simulates scoring code illegally rewriting a prediction

    with pytest.raises(LedgerImmutableError, match="does not match its recorded hash"):
        verify_ledger_integrity("run_z")
    with pytest.raises(LedgerImmutableError):
        read_prediction_ledger("run_z")


def test_scoring_does_not_mutate_the_real_phase4_ledger():
    """Integration check against the real Phase 4 ledger (if present): running the full
    scoring/comparison pipeline must leave the ledger file byte-for-byte unchanged."""
    from nfl_predict.backtesting.scoring import compare_to_baselines, score_holdout

    ledger_path, hash_path = ledger_paths("phase4_v1")
    if not ledger_path.is_file():
        pytest.skip("phase4_v1 ledger not present in this environment")

    before_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    results = score_holdout("phase4_v1")
    compare_to_baselines(results)

    after_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert before_hash == after_hash
    assert verify_ledger_integrity("phase4_v1") is True
