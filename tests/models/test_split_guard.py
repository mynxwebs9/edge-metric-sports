"""The sealed-holdout guard is the single most important safety property of Phase 3."""

from __future__ import annotations

import pytest

from nfl_predict.models.split import (
    ALL_PHASE3_VISIBLE_SEASONS,
    DEVELOPMENT_SEASONS,
    SEALED_HOLDOUT_SEASONS,
    VALIDATION_SEASON,
    SealedHoldoutError,
    assert_seasons_not_sealed,
)


def test_development_and_validation_seasons_do_not_overlap_sealed_holdout():
    assert not (set(DEVELOPMENT_SEASONS) & set(SEALED_HOLDOUT_SEASONS))
    assert VALIDATION_SEASON not in SEALED_HOLDOUT_SEASONS


def test_sealed_holdout_is_exactly_2024_2025():
    assert set(SEALED_HOLDOUT_SEASONS) == {2024, 2025}


def test_development_is_2010_through_2022():
    assert DEVELOPMENT_SEASONS == list(range(2010, 2023))


def test_validation_is_2023():
    assert VALIDATION_SEASON == 2023


def test_guard_allows_development_and_validation_seasons():
    assert_seasons_not_sealed(DEVELOPMENT_SEASONS)
    assert_seasons_not_sealed([VALIDATION_SEASON])
    assert_seasons_not_sealed(ALL_PHASE3_VISIBLE_SEASONS)


def test_guard_rejects_2024():
    with pytest.raises(SealedHoldoutError, match="2024"):
        assert_seasons_not_sealed([2024])


def test_guard_rejects_2025():
    with pytest.raises(SealedHoldoutError, match="2025"):
        assert_seasons_not_sealed([2025])


def test_guard_rejects_a_mixed_list_containing_a_sealed_season():
    with pytest.raises(SealedHoldoutError):
        assert_seasons_not_sealed([2022, 2023, 2024])


def test_load_game_dataset_refuses_sealed_seasons():
    from nfl_predict.models.feature_matrix import load_game_dataset

    with pytest.raises(SealedHoldoutError):
        load_game_dataset([2024])
    with pytest.raises(SealedHoldoutError):
        load_game_dataset([2025])
    with pytest.raises(SealedHoldoutError):
        load_game_dataset([2022, 2024])
