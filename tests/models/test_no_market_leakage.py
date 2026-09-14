"""Independent models must never see sportsbook/market fields."""

from __future__ import annotations

import pytest

from nfl_predict.features.registry import MARKET_FIELD_DENYLIST
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES, load_game_dataset

KNOWN_MARKET_FIELDS = {
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "spread_odds", "total_odds", "home_spread_odds", "away_spread_odds",
    "under_odds", "over_odds", "closing_line", "opening_line",
}


def test_no_ablation_feature_set_references_a_market_field():
    for name, cols in ABLATIONS.items():
        for col in cols:
            underlying = col.removeprefix("diff_").removeprefix("home_").removeprefix("away_")
            assert underlying not in MARKET_FIELD_DENYLIST, f"{name} references market field via {col}"
            assert underlying not in KNOWN_MARKET_FIELDS, f"{name} references market field via {col}"
    for col in CORE_FEATURES:
        underlying = col.removeprefix("diff_").removeprefix("home_").removeprefix("away_")
        assert underlying not in MARKET_FIELD_DENYLIST


def test_loaded_game_dataset_contains_no_market_fields():
    ds = load_game_dataset([2022])
    cols = set(ds.frame.columns)
    for field_name in KNOWN_MARKET_FIELDS | set(MARKET_FIELD_DENYLIST):
        assert field_name not in cols
        assert f"home_{field_name}" not in cols
        assert f"away_{field_name}" not in cols
        assert f"diff_{field_name}" not in cols


def test_guard_fires_if_a_market_field_were_added_to_an_ablation():
    """Defensive proof that the underlying guard (features.registry) actually rejects a
    market field, so a future accidental addition to an ABLATIONS list would fail loudly
    rather than silently training on it."""
    from nfl_predict.features.registry import assert_no_denylisted_columns

    with pytest.raises(ValueError, match="spread_line"):
        assert_no_denylisted_columns(["diff_off_epa_pp_season", "spread_line"])
