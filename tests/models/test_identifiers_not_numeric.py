"""Team/game identifiers must remain identifiers, never numeric predictive signals."""

from __future__ import annotations

import pytest

from nfl_predict.models.feature_matrix import (
    ABLATIONS,
    CORE_FEATURES,
    IDENTIFIER_COLUMN_NAMES,
    assert_no_identifier_columns,
    load_game_dataset,
    select_features,
)


def test_no_ablation_includes_an_identifier_column():
    for name, cols in ABLATIONS.items():
        assert not (set(cols) & IDENTIFIER_COLUMN_NAMES), f"{name} includes an identifier column"
    assert not (set(CORE_FEATURES) & IDENTIFIER_COLUMN_NAMES)


def test_guard_rejects_team_id_as_a_feature():
    with pytest.raises(ValueError, match="team_id"):
        assert_no_identifier_columns(["diff_off_epa_pp_season", "team_id"])


def test_guard_rejects_game_id_as_a_feature():
    with pytest.raises(ValueError, match="game_id"):
        assert_no_identifier_columns(["game_id"])


def test_select_features_refuses_identifier_columns():
    ds = load_game_dataset([2022])
    with pytest.raises(ValueError, match="home_team_id"):
        select_features(ds, ["diff_off_epa_pp_season", "home_team_id"])


def test_team_id_values_are_strings_not_numbers():
    """Sanity check on the raw data itself: team_id is a zero-padded string ("0200",
    "0325", ...) that would silently lose meaning (and collide/reorder) if ever cast to
    int - e.g. "0200" -> 200 loses the leading zero and implies an ordering that doesn't
    exist. Confirms the type actually is string in the source table, not just in our
    feature-selection guard."""
    import polars as pl

    ds = load_game_dataset([2022])
    # home_team_id/away_team_id aren't in the public feature-only frame after
    # select_features, but they ARE in the joined dataset frame (used for grouping/joins,
    # never as a model input) - confirm their dtype there.
    assert ds.frame.schema["home_team_id"] == pl.Utf8
    assert ds.frame.schema["away_team_id"] == pl.Utf8
    assert ds.frame.schema["game_id"] == pl.Utf8
