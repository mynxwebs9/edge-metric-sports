"""Phase 5 Step 23 proofs #9-10: the Phase 4 prediction ledger remains immutable, and
market scoring never mutates a model's stored prediction values - exercised against the
real end-to-end Phase 5 pipeline, not a mock."""

from __future__ import annotations

import hashlib

import polars as pl
import pytest

from nfl_predict.backtesting.ledger import ledger_paths, verify_ledger_integrity


def _skip_if_no_ledger():
    ledger_path, _ = ledger_paths("phase4_v1")
    if not ledger_path.is_file():
        pytest.skip("phase4_v1 ledger not present in this environment")
    return ledger_path


def test_running_the_full_market_benchmark_pipeline_leaves_the_ledger_byte_identical():
    ledger_path = _skip_if_no_ledger()
    before_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    before_bytes = ledger_path.read_bytes()

    from nfl_predict.backtesting.scoring import load_scored_frame
    from nfl_predict.market.ats import build_ats_bets, summarize_ats_bets
    from nfl_predict.market.benchmark import game_level_frame, run_market_benchmark
    from nfl_predict.market.moneyline_edge import build_moneyline_bets, summarize_moneyline_bets

    run_market_benchmark("phase4_v1")
    scored = load_scored_frame("phase4_v1")
    market_frame = game_level_frame("phase4_v1")

    preds = scored.filter((pl.col("model_id") == "elo_v2") & (pl.col("target") == "home_margin")).select(["game_id", "predicted_value"])
    margin_frame = market_frame.join(preds, on="game_id", how="inner")
    bets = build_ats_bets("elo_v2", margin_frame["predicted_value"].to_numpy(), margin_frame)
    summarize_ats_bets(bets)

    win_preds = scored.filter((pl.col("model_id") == "elo_v2") & (pl.col("target") == "home_win")).select(["game_id", "predicted_value"])
    win_frame = market_frame.join(win_preds, on="game_id", how="inner")
    ml_bets = build_moneyline_bets("elo_v2", win_frame["predicted_value"].to_numpy(), win_frame)
    summarize_moneyline_bets(ml_bets)

    after_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    after_bytes = ledger_path.read_bytes()
    assert before_hash == after_hash
    assert before_bytes == after_bytes
    assert verify_ledger_integrity("phase4_v1") is True


def test_a_models_predicted_values_are_identical_before_and_after_market_scoring():
    _skip_if_no_ledger()
    from nfl_predict.backtesting.scoring import load_scored_frame
    from nfl_predict.market.benchmark import run_market_benchmark

    before = load_scored_frame("phase4_v1").filter(pl.col("model_id") == "elo_v2").sort(["game_id", "target"])
    before_values = before["predicted_value"].to_list()

    run_market_benchmark("phase4_v1")

    after = load_scored_frame("phase4_v1").filter(pl.col("model_id") == "elo_v2").sort(["game_id", "target"])
    after_values = after["predicted_value"].to_list()

    assert before_values == after_values
