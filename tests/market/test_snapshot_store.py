"""Phase 5 Step 23 proofs #8, #11: market data lives in its own layer (never the football
feature tables) and missing market prices are never silently assumed."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predict.market.snapshot_store import (
    MARKET_SNAPSHOT_COLUMNS,
    build_market_snapshot_for_season,
    read_market_snapshot,
)


def test_market_snapshot_covers_every_2024_and_2025_holdout_game():
    """Integration check against the real Phase 4 holdout ledger (if present)."""
    from nfl_predict.backtesting.ledger import ledger_paths, read_prediction_ledger

    ledger_path, _ = ledger_paths("phase4_v1")
    if not ledger_path.is_file():
        pytest.skip("phase4_v1 ledger not present in this environment")

    ledger = read_prediction_ledger("phase4_v1")
    market = read_market_snapshot([2024, 2025])
    holdout_game_ids = ledger.select("game_id").unique()
    missing = holdout_game_ids.join(market.select("game_id").unique(), on="game_id", how="anti")
    assert missing.height == 0, f"{missing.height} holdout games have no market_snapshot row"


def test_market_snapshot_has_zero_nulls_in_core_price_fields_for_a_real_season():
    frame = build_market_snapshot_for_season(2024)
    core_cols = [
        "home_spread_traditional", "away_spread_traditional", "home_spread_price", "away_spread_price",
        "home_moneyline", "away_moneyline", "total_line", "over_price", "under_price",
    ]
    nulls = frame.select(core_cols).null_count()
    for col in core_cols:
        assert nulls[col][0] == 0, f"unexpected null in {col} for season 2024"


def test_market_snapshot_never_labels_the_single_nflverse_line_as_closing():
    frame = build_market_snapshot_for_season(2024)
    assert set(frame["snapshot_type"].unique().to_list()) == {"UNKNOWN"}
    assert frame["sportsbook"].null_count() == frame.height  # unattributed, not fabricated


def test_market_snapshot_spread_sign_matches_moneyline_favorite_direction():
    """Cross-check: whichever side has the negative (favorite) moneyline must also have the
    negative (favorite) traditional spread - real data, not a synthetic fixture."""
    frame = build_market_snapshot_for_season(2024)
    home_favored_by_ml = frame["home_moneyline"].to_numpy() < 0
    home_favored_by_spread = frame["home_spread_traditional"].to_numpy() < 0
    agree = (home_favored_by_ml == home_favored_by_spread).mean()
    assert agree > 0.95  # allow for the rare near-even-money game where signs can legitimately differ


def test_build_market_snapshot_raises_rather_than_assumes_for_an_uningested_season():
    with pytest.raises(ValueError, match="No canonical"):
        build_market_snapshot_for_season(1899)


def test_read_market_snapshot_returns_empty_frame_not_a_fabricated_one_when_nothing_written(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nfl_predict.market.snapshot_store.get_settings",
        lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})(),
    )
    frame = read_market_snapshot([1899])
    assert frame.height == 0
    assert set(MARKET_SNAPSHOT_COLUMNS) <= set(frame.columns)


def test_market_module_is_never_imported_by_the_independent_model_or_feature_code():
    """Structural proof that market data cannot enter the independent model: neither
    nfl_predict.models nor nfl_predict.features imports nfl_predict.market anywhere."""
    import ast
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[2] / "src" / "nfl_predict"
    offending = []
    for package in ("models", "features"):
        for path in (src_root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module else []
                else:
                    continue
                if any(n and n.startswith("nfl_predict.market") for n in names):
                    offending.append(str(path))
    assert not offending, f"nfl_predict.market imported from independent-model code: {offending}"
