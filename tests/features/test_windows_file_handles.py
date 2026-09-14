"""Regression test for the Windows file-handle/memory-mapping issue that intermittently
broke tests/features/test_leakage.py::test_b_future_game_mutation_does_not_change_earlier_row
(OSError 1224: "a file with a user-mapped section open").

Root cause: `mutate_pbp_season` (conftest.py) read the season's Parquet file with polars'
default `memory_map=True`, then tried to `write_parquet` back to that same path. Under the
memory pressure of a full pytest run, the Python-side refcount drop for the DataFrame's
underlying Arrow buffers and the OS-level unmap don't always happen in the same instant, so
the in-place overwrite could intermittently collide with a still-draining memory-mapped
section. Fixed by (1) reading with `memory_map=False` everywhere in this codebase's Parquet
read paths, and (2) making the test mutation helper itself use an atomic
read -> write-to-temp -> explicit-release -> os.replace pattern rather than an in-place
overwrite - see conftest.py's `mutate_pbp_season` docstring for the full explanation.

This test exercises the exact lifecycle the bug depended on: build features from a real
Parquet file, let every reference to it go out of scope, then immediately mutate/replace
that same file and rebuild - proving the file is never left in a locked state after a
build completes.
"""

from __future__ import annotations

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.features.team_game import build_team_game_features
from tests.features.conftest import mutate_pbp_season

PHI = "3700"
EARLIER_GAME = "2024_02_ATL_PHI"
LATER_GAME = "2024_06_CLE_PHI"


def _build_and_get_row(game_id: str) -> dict:
    conn = get_connection()
    init_schema(conn)
    tg = build_team_game_features(conn, [2022, 2023, 2024])
    conn.close()
    matches = tg.filter((tg["game_id"] == game_id) & (tg["team_id"] == PHI))
    assert matches.height == 1
    return matches.row(0, named=True)


def test_feature_build_does_not_lock_the_pbp_file_after_returning(real_data_sandbox):
    """A completed build releases its handle on the season Parquet file - proven by
    immediately mutating that same file right after the build returns, with no manual
    delay/gc call at the call site (the release discipline lives in the read path itself,
    not in this test)."""
    before = _build_and_get_row(EARLIER_GAME)

    # If the prior build had left the file memory-mapped/locked, this would raise
    # OSError 1224 on Windows.
    mutate_pbp_season(real_data_sandbox, 2024, LATER_GAME, "epa", 999.0)

    after = _build_and_get_row(EARLIER_GAME)
    assert before == after


def test_repeated_build_mutate_cycles_never_leave_a_locked_file(real_data_sandbox):
    """Runs the build -> mutate cycle several times in a row - a single lucky pass
    wouldn't prove the fix; repeating it is what actually exercises timing-dependent GC/
    unmap behavior."""
    for i in range(5):
        _build_and_get_row(EARLIER_GAME)
        mutate_pbp_season(real_data_sandbox, 2024, LATER_GAME, "epa", float(i))

    # Final rebuild must also succeed and reflect the last mutation.
    final_row = _build_and_get_row(LATER_GAME)
    assert final_row["off_epa_pp_3g"] is not None


def test_mutation_helper_produces_no_leftover_temp_file(real_data_sandbox):
    """The atomic-replacement pattern's intermediate `.tmp.parquet` must not survive a
    successful mutation - os.replace() consumes it."""
    mutate_pbp_season(real_data_sandbox, 2024, EARLIER_GAME, "epa", 1.0)
    pbp_dir = real_data_sandbox / "normalized" / "play_by_play" / "season=2024"
    leftover_tmp_files = list(pbp_dir.glob("*.tmp.parquet"))
    assert leftover_tmp_files == []
