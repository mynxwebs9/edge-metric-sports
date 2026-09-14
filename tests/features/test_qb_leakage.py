"""Item 10 of the Phase 2 continuation: dedicated review of QB continuity for leakage.

Proves, with a synthetic but realistic QB-change scenario, that:
- A game's QB features describe the PREVIOUS game's primary QB, never this game's own.
- Mutating a game's own starter/stats never changes an EARLIER game's QB features.
- A genuine QB change is picked up starting the game AFTER it happens, not the game it
  happens in (since that would require using the target game's own starter identity).

See docs/PHASE2_FEATURE_REPORT.md#qb-features-and-the-leakage-rule for the written argument
this test automates.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.features.qb import build_qb_atomic, build_qb_features, identify_primary_qb

QB_X = "00-QBX"
QB_Y = "00-QBY"


def _play(game_id, team_id, passer, epa=0.5, success=1, pass_attempt=1, sack=0, interception=0, scramble=0, rusher=None):
    return dict(
        game_id=game_id, posteam_id=team_id, qb_dropback=1, epa=epa, success=success,
        pass_attempt=pass_attempt, sack=sack, interception=interception, cpoe=1.0,
        qb_scramble=scramble, passer_player_id=passer, rusher_player_id=rusher,
    )


def _synthetic_pbp() -> pl.DataFrame:
    rows = []
    # Game 1: QB_X starts and plays every dropback.
    rows += [_play("G1", "A", QB_X) for _ in range(30)]
    # Game 2: QB_X gets hurt after a few plays; QB_Y takes over and throws the most.
    rows += [_play("G2", "A", QB_X) for _ in range(5)]
    rows += [_play("G2", "A", QB_Y, epa=1.0) for _ in range(25)]
    # Game 3: QB_Y starts and plays every dropback.
    rows += [_play("G3", "A", QB_Y, epa=2.0) for _ in range(30)]
    return pl.DataFrame(rows)


def _team_game_index() -> pl.DataFrame:
    return pl.DataFrame({
        "team_id": ["A", "A", "A"],
        "season": [2024, 2024, 2024],
        "sort_ts": ["2024-09-01", "2024-09-08", "2024-09-15"],
        "game_id": ["G1", "G2", "G3"],
    })


def test_qb_change_is_reflected_starting_the_game_after_it_happens():
    pbp = _synthetic_pbp()
    qb_atomic = build_qb_atomic(pbp)
    primary = identify_primary_qb(qb_atomic)

    # Sanity: game 2's OWN primary QB (most dropbacks) is QB_Y, not QB_X.
    g2_primary = primary.filter(pl.col("game_id") == "G2").row(0, named=True)["primary_qb_id"]
    assert g2_primary == QB_Y

    result = build_qb_features(_team_game_index(), qb_atomic, primary, min_observations=1)
    rows = {r["game_id"]: r for r in result.to_dicts()}

    # Game 1: no prior game -> no primary QB identified yet.
    assert rows["G1"]["qb_primary_id"] is None

    # Game 2: describes game 1's primary QB (QB_X) - NOT game 2's own primary (QB_Y), even
    # though QB_Y is who actually played most of game 2. This is the crux of the leakage
    # rule: game 2's PREGAME QB feature cannot know that QB_Y would take over mid-game.
    assert rows["G2"]["qb_primary_id"] == QB_X

    # Game 3: describes game 2's primary QB (QB_Y) - the change is now visible, one game
    # after it actually happened, using only fully-completed information.
    assert rows["G3"]["qb_primary_id"] == QB_Y
    assert rows["G3"]["qb_epa_dropback_season"] == 1.0  # QB_Y's own EPA/dropback in game 2
    assert rows["G3"]["qb_starts_season"] == 1


def test_mutating_a_later_games_qb_does_not_change_an_earlier_games_qb_features():
    pbp = _synthetic_pbp()
    qb_atomic = build_qb_atomic(pbp)
    primary = identify_primary_qb(qb_atomic)
    baseline = {r["game_id"]: r for r in build_qb_features(_team_game_index(), qb_atomic, primary).to_dicts()}

    # Mutate game 3 heavily (a different QB entirely, wildly different EPA).
    mutated_pbp = pl.concat([
        pbp.filter(pl.col("game_id") != "G3"),
        pl.DataFrame([_play("G3", "A", "00-QBZ", epa=-99.0) for _ in range(30)]),
    ])
    qb_atomic_2 = build_qb_atomic(mutated_pbp)
    primary_2 = identify_primary_qb(qb_atomic_2)
    mutated = {r["game_id"]: r for r in build_qb_features(_team_game_index(), qb_atomic_2, primary_2).to_dicts()}

    # Games 1 and 2's QB features must be completely unaffected by game 3's mutation.
    for col in ("qb_primary_id", "qb_epa_dropback_season", "qb_starts_season", "qb_consecutive_starts"):
        assert baseline["G1"][col] == mutated["G1"][col], f"G1.{col} changed after mutating G3"
        assert baseline["G2"][col] == mutated["G2"][col], f"G2.{col} changed after mutating G3"


def test_qb_primary_id_is_never_derived_from_the_target_games_own_plays():
    """Directly asserts the leakage-critical property using the one row in this fixture
    where "this game's own primary QB" and "the previous game's primary QB" DIFFER (game 2:
    own primary is QB_Y, previous-game primary is QB_X) - if build_qb_features ever leaked
    the target game's own starter identity, game 2's qb_primary_id would come back QB_Y
    instead of QB_X."""
    pbp = _synthetic_pbp()
    qb_atomic = build_qb_atomic(pbp)
    primary = identify_primary_qb(qb_atomic)
    own_primary_by_game = {r["game_id"]: r["primary_qb_id"] for r in primary.to_dicts()}

    result = build_qb_features(_team_game_index(), qb_atomic, primary)
    rows = {r["game_id"]: r for r in result.to_dicts()}

    assert own_primary_by_game["G2"] == QB_Y  # sanity: G2's own primary really is QB_Y
    assert rows["G2"]["qb_primary_id"] != own_primary_by_game["G2"]
    assert rows["G2"]["qb_primary_id"] == QB_X
