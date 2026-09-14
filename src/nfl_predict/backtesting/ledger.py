"""Phase 4 Step 3: the immutable prediction ledger.

Every prediction the walk-forward protocol produces is written here, in full, BEFORE any
outcome is ever joined to it (see `scoring.py`, which only ever READS this ledger - it never
regenerates or edits a prediction after seeing how the game turned out). Once written, a
run's ledger file is never edited in place - a SHA-256 sidecar hash (the same tamper-evidence
pattern as the Phase 4 holdout-freeze manifest) makes any later mutation detectable, and
`write_prediction_ledger` refuses to overwrite an existing run's file at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.backtesting.walk_forward import PredictionRecord

LEDGER_DIRNAME = "backtests/predictions"
LEDGER_FILENAME = "predictions.parquet"
LEDGER_HASH_FILENAME = "predictions.sha256"

LEDGER_COLUMNS = [
    "game_id", "season", "week", "season_type", "kickoff_timestamp", "prediction_generated_at_utc",
    "model_id", "model_version", "feature_version", "target", "predicted_value",
    "training_cutoff_season", "training_cutoff_week", "training_row_count",
    "artifact_hash", "feature_names_hash", "run_id",
]


class LedgerImmutableError(Exception):
    """Raised when code tries to overwrite an existing run's prediction ledger, or when a
    ledger's content no longer matches its recorded hash (tampered after being written)."""


@dataclass(frozen=True)
class LedgerWriteResult:
    ledger_path: Path
    hash_path: Path
    ledger_sha256: str
    n_predictions: int


def ledger_paths(run_id: str) -> tuple[Path, Path]:
    directory = get_settings().data_dir / LEDGER_DIRNAME / f"run_id={run_id}"
    return directory / LEDGER_FILENAME, directory / LEDGER_HASH_FILENAME


def write_prediction_ledger(records: list[PredictionRecord], run_id: str) -> LedgerWriteResult:
    ledger_path, hash_path = ledger_paths(run_id)
    if ledger_path.is_file():
        raise LedgerImmutableError(
            f"A prediction ledger already exists for run_id={run_id!r} at {ledger_path}. "
            "Predictions are immutable snapshots once written (CLAUDE.md principle 4) - "
            "use a new run_id for a new backtest run rather than overwriting this one."
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in records:
        row = asdict(r)
        row["prediction_generated_at_utc"] = generated_at
        row["run_id"] = run_id
        rows.append(row)

    # Explicit schema, not inferred: some columns are legitimately null for some models
    # (Elo/naive have no `artifact_hash`/`feature_names_hash`, some games have no
    # `kickoff_timestamp`) - polars' schema inference only samples the first N rows and can
    # pick the wrong dtype (or choke outright) when a later row's non-null value doesn't fit
    # what it guessed from a null-heavy prefix.
    schema = {
        "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "season_type": pl.Utf8,
        "kickoff_timestamp": pl.Utf8, "prediction_generated_at_utc": pl.Utf8,
        "model_id": pl.Utf8, "model_version": pl.Utf8, "feature_version": pl.Utf8,
        "target": pl.Utf8, "predicted_value": pl.Float64,
        "training_cutoff_season": pl.Int64, "training_cutoff_week": pl.Int64, "training_row_count": pl.Int64,
        "artifact_hash": pl.Utf8, "feature_names_hash": pl.Utf8, "run_id": pl.Utf8,
    }
    frame = pl.DataFrame(rows, schema=schema).select(LEDGER_COLUMNS) if rows else pl.DataFrame(schema=schema)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(ledger_path)

    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    hash_path.write_text(digest, encoding="utf-8")

    return LedgerWriteResult(ledger_path=ledger_path, hash_path=hash_path, ledger_sha256=digest, n_predictions=frame.height)


def verify_ledger_integrity(run_id: str) -> bool:
    ledger_path, hash_path = ledger_paths(run_id)
    if not ledger_path.is_file() or not hash_path.is_file():
        raise LedgerImmutableError(f"No prediction ledger found for run_id={run_id!r} at {ledger_path}")
    actual = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    recorded = hash_path.read_text(encoding="utf-8").strip()
    if actual != recorded:
        raise LedgerImmutableError(
            f"Prediction ledger for run_id={run_id!r} does not match its recorded hash "
            f"({actual} != {recorded}) - it was modified after being written."
        )
    return True


def read_prediction_ledger(run_id: str) -> pl.DataFrame:
    """Read-only access to an already-written, integrity-verified ledger. Every consumer
    (scoring, error analysis, reporting) must go through this - never read the parquet file
    directly - so a tampered ledger is always caught before it can silently affect results."""
    verify_ledger_integrity(run_id)
    ledger_path, _ = ledger_paths(run_id)
    return pl.read_parquet(ledger_path, memory_map=False)
