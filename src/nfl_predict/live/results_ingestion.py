"""Phase 8A Step 20: results ingestion.

Reads final outcomes straight from the `games` table (already updated by re-running
`nfl_predict.data.ingest`) and writes them to a SEPARATE, immutable "graded results" store -
never touching the original pregame prediction (`prediction_publication.py`), market
(`nfl_predict.market.live_snapshot_store`), or research (`nfl_predict.research.storage`)
records. `compute_graded_result` returns `None` (never a fabricated outcome) for a game that
is not yet `final`. Elo's sequential rating state needs no separate "update" step here - it
is recomputed fresh from the `games` table on every call to
`nfl_predict.live.live_models.predict_live_elo`, which naturally reflects any newly
completed games the next time it runs (see that module).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nfl_predict.config import get_settings
from nfl_predict.data.db import get_connection, init_schema

GRADED_RESULTS_DIRNAME = "live/graded_results"


class ConflictingGradedResultError(Exception):
    """Raised when a DIFFERENT graded result already exists for a game_id - results are
    immutable once written; re-submitting the identical result is a safe no-op."""


@dataclass(frozen=True)
class GradedResult:
    game_id: str
    season: int
    week: int
    home_score: int
    away_score: int
    home_margin: int
    total_points: int
    home_win: bool | None  # None for a tie
    is_tie: bool
    graded_at: str


def compute_graded_result(game_id: str) -> GradedResult | None:
    """Returns None if the game is not yet `final` - never fabricates an outcome for a
    game still in progress or not yet played."""
    conn = get_connection()
    init_schema(conn)
    try:
        row = conn.execute(
            "SELECT season, week, home_score, away_score, game_status FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row["game_status"] != "final":
        return None

    home_score, away_score = row["home_score"], row["away_score"]
    is_tie = home_score == away_score
    return GradedResult(
        game_id=game_id, season=row["season"], week=row["week"], home_score=home_score, away_score=away_score,
        home_margin=home_score - away_score, total_points=home_score + away_score,
        home_win=None if is_tie else bool(home_score > away_score), is_tie=is_tie,
        graded_at=datetime.now(timezone.utc).isoformat(),
    )


def _paths(game_id: str) -> tuple[Path, Path]:
    directory = get_settings().data_dir / GRADED_RESULTS_DIRNAME
    return directory / f"{game_id}.json", directory / f"{game_id}.sha256"


def write_graded_result(result: GradedResult) -> Path:
    data_path, hash_path = _paths(result.game_id)
    if data_path.is_file():
        existing = json.loads(data_path.read_bytes().decode("utf-8"))
        new_payload = {k: v for k, v in asdict(result).items() if k != "graded_at"}
        existing_payload = {k: v for k, v in existing.items() if k != "graded_at"}
        if existing_payload == new_payload:
            return data_path  # identical result already recorded - idempotent no-op
        raise ConflictingGradedResultError(
            f"A DIFFERENT graded result already exists for game_id={result.game_id!r} - "
            "graded results are immutable once written."
        )
    encoded = json.dumps(asdict(result), indent=2, sort_keys=True, default=str).encode("utf-8")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(encoded)
    hash_path.write_text(hashlib.sha256(encoded).hexdigest(), encoding="utf-8")
    return data_path


def read_graded_result(game_id: str) -> dict | None:
    data_path, hash_path = _paths(game_id)
    if not data_path.is_file():
        return None
    content = data_path.read_bytes()
    if hash_path.is_file():
        actual = hashlib.sha256(content).hexdigest()
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if actual != recorded:
            raise ValueError(f"Graded result for {game_id!r} does not match its recorded hash - modified after being written.")
    return json.loads(content.decode("utf-8"))
