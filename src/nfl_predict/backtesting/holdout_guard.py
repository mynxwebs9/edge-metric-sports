"""The Phase 4 unsealing gate.

Nothing in this codebase may read 2024-2025 target outcomes or build a holdout feature
dataset UNLESS `assert_freeze_complete()` succeeds - which requires the freeze manifest
(`data/backtests/phase4_holdout_freeze.json`) to exist AND its recorded SHA-256 sidecar to
match the manifest's current content exactly. This is deliberately a re-hash-and-compare
check, not a bare existence check: if the manifest were edited after freezing (e.g. to add
a model or change a hyperparameter after peeking at results), the hash would no longer
match and every holdout-reading function in this module raises.

This module is the ONLY place 2024/2025 target Parquet files get written
(`unseal_and_write_holdout_targets`) and the ONLY place the seal-bypassing internal feature
loader (`nfl_predict.models.feature_matrix._load_game_dataset_impl`) is called for sealed
seasons. `nfl_predict.models.feature_matrix.load_game_dataset` and
`nfl_predict.models.split.assert_seasons_not_sealed` are completely unchanged and still
unconditionally refuse 2024/2025 - Phase 3's guarantee stays intact for Phase 3 code;
Phase 4 code goes through this separate, explicitly-gated door instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.models.feature_matrix import ModelDataset, _load_game_dataset_impl
from nfl_predict.models.split import SEALED_HOLDOUT_SEASONS
from nfl_predict.models.targets import write_targets

from nfl_predict.backtesting.freeze import (
    CANDIDATE_MODELS,
    freeze_manifest_paths,
)

REQUIRED_MODEL_IDS = frozenset(c["model_id"] for c in CANDIDATE_MODELS)


class FreezeNotCompleteError(Exception):
    """Raised when Phase 4 code tries to touch the sealed holdout before the freeze
    manifest exists, is unmodified since it was written, or is missing a required
    candidate model."""


def assert_freeze_complete() -> dict:
    manifest_path, hash_path = freeze_manifest_paths()
    if not manifest_path.is_file() or not hash_path.is_file():
        raise FreezeNotCompleteError(
            f"Holdout freeze manifest not found at {manifest_path} (or its hash sidecar "
            f"{hash_path}). Run the Phase 4 freeze step "
            "(nfl_predict.backtesting.freeze.write_freeze_manifest) BEFORE touching "
            "2024-2025 - see docs/PHASE4_BACKTEST_REPORT.md."
        )

    content = manifest_path.read_text(encoding="utf-8")
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    recorded_hash = hash_path.read_text(encoding="utf-8").strip()
    if actual_hash != recorded_hash:
        raise FreezeNotCompleteError(
            f"Freeze manifest at {manifest_path} does not match its recorded hash "
            f"({actual_hash} != {recorded_hash}) - it was modified after being frozen. "
            "Refusing to unseal the holdout against a tampered manifest."
        )

    manifest = json.loads(content)
    present_ids = {c["model_id"] for c in manifest.get("candidate_models", [])}
    missing = REQUIRED_MODEL_IDS - present_ids
    if missing:
        raise FreezeNotCompleteError(f"Freeze manifest is missing required candidate model(s): {sorted(missing)}")

    return manifest


def unseal_and_write_holdout_targets() -> None:
    """The ONLY function in this codebase that writes 2024/2025 target Parquet files.
    Requires a complete, verified freeze first."""
    assert_freeze_complete()
    conn = get_connection()
    init_schema(conn)
    try:
        write_targets(conn, SEALED_HOLDOUT_SEASONS)
    finally:
        conn.close()


def load_holdout_game_dataset(seasons: list[int]) -> ModelDataset:
    """The ONLY function that loads a game-level feature+target dataset for sealed
    seasons. Requires a complete, verified freeze first, every time it's called - not just
    once at import time - so a manifest tampered with mid-session is still caught."""
    assert_freeze_complete()
    return _load_game_dataset_impl(seasons)
