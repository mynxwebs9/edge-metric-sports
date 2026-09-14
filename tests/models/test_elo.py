"""Elo: sequential update discipline, no look-ahead, home advantage explicit."""

from __future__ import annotations

import polars as pl

from nfl_predict.models.elo import EloConfig, EloModel


def _synthetic_games() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["G1", "G2", "G3", "G4"],
        "season": [2020, 2020, 2020, 2020],
        "week": [1, 2, 3, 4],
        "home_team_id": ["A", "B", "A", "C"],
        "away_team_id": ["B", "C", "C", "A"],
        "home_margin": [10, -3, 7, 0],
        "home_win": [1, 0, 1, None],
        "is_tie": [False, False, False, True],
    })


def test_ratings_start_at_initial_value():
    model = EloModel(EloConfig(k_factor=20, home_field_advantage=50))
    history = model.run_sequential(_synthetic_games())
    row1 = history.row(0, named=True)
    assert row1["home_elo_pre"] == 1500.0
    assert row1["away_elo_pre"] == 1500.0


def test_home_advantage_is_explicit_and_affects_expected_probability():
    games = _synthetic_games().head(1)
    no_adv = EloModel(EloConfig(k_factor=20, home_field_advantage=0)).run_sequential(games)
    with_adv = EloModel(EloConfig(k_factor=20, home_field_advantage=65)).run_sequential(games)
    assert with_adv.row(0, named=True)["expected_home_win_prob"] > no_adv.row(0, named=True)["expected_home_win_prob"]


def test_ratings_only_update_after_the_game_and_never_affect_earlier_games():
    """The pre-game rating recorded for game N must be identical whether or not games
    AFTER N have happened yet - i.e. running the model game-by-game one at a time produces
    the same pre-game ratings as running it on the full sequence at once."""
    games = _synthetic_games()
    full_run = EloModel(EloConfig(k_factor=20, home_field_advantage=50)).run_sequential(games)

    model = EloModel(EloConfig(k_factor=20, home_field_advantage=50))
    incremental_rows = []
    for i in range(games.height):
        incremental_rows.append(model.run_sequential(games.slice(i, 1)).row(0, named=True))

    for a, b in zip(full_run.iter_rows(named=True), incremental_rows):
        assert a["home_elo_pre"] == b["home_elo_pre"]
        assert a["away_elo_pre"] == b["away_elo_pre"]
        assert a["expected_home_win_prob"] == b["expected_home_win_prob"]


def test_appending_a_future_unrelated_game_does_not_change_earlier_ratings():
    """A game added AFTER game 4 (even one that doesn't involve any of A/B/C) must not
    change any of games 1-4's already-recorded PRE-game ratings. Note this is a different,
    narrower claim than "a team's rating only reflects its own games" - that claim is false
    for Elo by construction, since a team's rating is relational: an opponent's OTHER
    results (from games this team isn't even part of) legitimately feed into what that
    opponent's rating is when the two later meet. That's expected Elo behavior, not
    leakage - what must never happen is a LATER game changing an EARLIER prediction."""
    games = _synthetic_games()
    baseline = EloModel(EloConfig(k_factor=20, home_field_advantage=50)).run_sequential(games)

    games_with_future_addition = pl.concat([
        games,
        pl.DataFrame({
            "game_id": ["G5"], "season": [2020], "week": [5],
            "home_team_id": ["D"], "away_team_id": ["E"],
            "home_margin": [100], "home_win": [1], "is_tie": [False],
        }),
    ])
    extended = EloModel(EloConfig(k_factor=20, home_field_advantage=50)).run_sequential(games_with_future_addition)

    for game_id in ("G1", "G2", "G3", "G4"):
        b = baseline.filter(pl.col("game_id") == game_id).row(0, named=True)
        e = extended.filter(pl.col("game_id") == game_id).row(0, named=True)
        assert b["home_elo_pre"] == e["home_elo_pre"]
        assert b["away_elo_pre"] == e["away_elo_pre"]
        assert b["expected_home_win_prob"] == e["expected_home_win_prob"]


def test_tie_updates_ratings_toward_convergence_not_a_win_or_loss():
    games = _synthetic_games().filter(pl.col("game_id") == "G4")  # the tie: home=C, away=A
    model = EloModel(EloConfig(k_factor=20, home_field_advantage=0))
    model.run_sequential(games)
    # A tie against evenly-matched (both at 1500) teams should leave both ratings
    # approximately unchanged, since expected == 0.5 == actual outcome for a tie.
    assert abs(model.ratings["C"] - 1500.0) < 1e-9
    assert abs(model.ratings["A"] - 1500.0) < 1e-9


def test_season_regression_reverts_ratings_toward_the_mean_at_season_boundary():
    games = pl.DataFrame({
        "game_id": ["G1", "G2"], "season": [2020, 2021], "week": [1, 1],
        "home_team_id": ["A", "A"], "away_team_id": ["B", "B"],
        "home_margin": [20, 0], "home_win": [1, None], "is_tie": [False, True],
    })
    model = EloModel(EloConfig(k_factor=20, home_field_advantage=0, season_regression_fraction=1 / 3))
    history = model.run_sequential(games)
    rating_after_g1 = None
    # Recover A's rating right after G1 by re-running just G1.
    m2 = EloModel(EloConfig(k_factor=20, home_field_advantage=0))
    m2.run_sequential(games.filter(pl.col("game_id") == "G1"))
    rating_after_g1 = m2.ratings["A"]

    g2_row = history.filter(pl.col("game_id") == "G2").row(0, named=True)
    expected_after_regression = 1500.0 + (1 - 1 / 3) * (rating_after_g1 - 1500.0)
    assert abs(g2_row["home_elo_pre"] - expected_after_regression) < 1e-9
    assert g2_row["home_elo_pre"] != rating_after_g1  # regression actually happened
