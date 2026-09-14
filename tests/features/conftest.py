"""Shared fixtures for the Phase 2 feature-engine test suite.

`real_data_sandbox` copies a slice of the ALREADY-INGESTED, Phase-1-validated local dataset
(SQLite games/teams tables, play-by-play Parquet, raw snap_counts/schedules snapshots) into
an isolated tmp directory, then points NFL_DATA_DIR at it. This gives leakage/quality tests
a realistic dataset they can freely mutate (to prove leakage guards) without ever touching
the real project data/ directory, and without any network access - everything copied is
already a local file from Phase 1's historical backfill.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nfl_predict.config import PROJECT_ROOT

SANDBOX_SEASONS = [2022, 2023, 2024]


@pytest.fixture
def real_data_sandbox(tmp_path, monkeypatch):
    real_data_dir = PROJECT_ROOT / "data"
    if not (real_data_dir / "nfl_predict.sqlite").is_file():
        pytest.skip("Real ingested data (data/nfl_predict.sqlite) not present - run Phase 1 ingestion first.")

    shutil.copy(real_data_dir / "nfl_predict.sqlite", tmp_path / "nfl_predict.sqlite")

    for season in SANDBOX_SEASONS:
        src_pbp = real_data_dir / "normalized" / "play_by_play" / f"season={season}"
        if src_pbp.is_dir():
            shutil.copytree(src_pbp, tmp_path / "normalized" / "play_by_play" / f"season={season}")

    for dataset in ("snap_counts", "schedules"):
        src = real_data_dir / "raw" / "nflverse" / dataset
        if src.is_dir():
            shutil.copytree(src, tmp_path / "raw" / "nflverse" / dataset)

    monkeypatch.setenv("NFL_DATA_DIR", str(tmp_path))
    return tmp_path


def mutate_pbp_season(sandbox_dir: Path, season: int, game_id: str, column: str, value) -> None:
    """Overwrites `column` to `value` for every play belonging to `game_id`, in the
    sandbox's copy of that season's play-by-play Parquet. Used by leakage tests to prove a
    mutated game does/doesn't propagate where it should/shouldn't.

    Windows note: polars memory-maps Parquet reads by default (`memory_map=True`), which
    holds an open mapped section on the file for as long as anything still references the
    DataFrame's underlying Arrow buffers - and `with_columns` on an UNCHANGED column can
    still share those buffers rather than copying them. Under the memory pressure of a full
    pytest run (many other DataFrames alive at once), the Python-side refcount drop and the
    OS-level unmap don't always happen in the same instant, so a plain
    `read -> transform -> write_parquet(same path)` can intermittently fail with
    "OSError 1224: a file with a user-mapped section open" - it reproduced inconsistently
    even in isolated repros here, which is the signature of a timing-dependent handle-
    lifetime bug, not a logic bug. Fixed with an atomic-replacement pattern: read with
    `memory_map=False` (so this read never mmaps the file to begin with), write the
    mutated result to a NEW temporary path, explicitly drop every in-memory reference and
    force garbage collection (so nothing can still be holding the old buffers), THEN
    `os.replace()` the temp file over the original - a rename, which Windows allows even if
    a stale handle to the old inode is still draining, unlike an in-place overwrite.
    """
    import gc
    import os

    import polars as pl

    path = sandbox_dir / "normalized" / "play_by_play" / f"season={season}" / "data.parquet"
    source = pl.read_parquet(path, memory_map=False)
    mutated = source.with_columns(
        pl.when(pl.col("game_id") == game_id).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )

    tmp_path = path.with_suffix(".tmp.parquet")
    mutated.write_parquet(tmp_path)

    del source
    del mutated
    gc.collect()

    os.replace(tmp_path, path)
