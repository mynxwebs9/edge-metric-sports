"""Phase 8A Steps 15-16: multi-run game lifecycle and current-publishable-snapshot
selection.

Every pregame run for a game (Thursday initial, Friday injury update, Sunday-morning odds/
research update, T-minus-90-minutes final) is written as its own immutable, hash-verified
record - `write_pregame_run` raises `PregameRunAlreadyExistsError` on a duplicate run_id,
never overwrites. `select_latest_valid_pregame_snapshot` picks, from every preserved run for
a game, the single most recent one that: occurred strictly before kickoff (a run whose own
timestamp is at/after kickoff is NEVER eligible - Step 3/18's "post-kickoff data cannot
become a pregame snapshot"); has the required model output(s); has market data no older
than a configured maximum age; and meets whatever research-availability/freshness the
caller requires. Every older (or invalid) run remains on disk, untouched, for audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nfl_predict.config import get_settings
from nfl_predict.decision.staleness import compute_age_seconds, is_stale

PREGAME_RUN_DIRNAME = "live/pregame_runs"


class PregameRunAlreadyExistsError(Exception):
    """Raised when a (game_id, run_id) pregame run already exists - runs are immutable."""


@dataclass(frozen=True)
class PregameRunRecord:
    game_id: str
    season: int
    week: int
    run_id: str
    run_timestamp: str
    kickoff_timestamp: str | None
    elo_available: bool
    ridge_available: bool
    lightgbm_available: bool
    logistic_available: bool
    market_available: bool
    market_snapshot_timestamp: str | None
    research_available: bool
    research_classification: str | None
    research_timestamp: str | None
    payload: dict  # the full live packet for this run - opaque to this module


def _paths(season: int, week: int, game_id: str, run_id: str) -> tuple[Path, Path]:
    directory = get_settings().data_dir / PREGAME_RUN_DIRNAME / f"season={season}" / f"week={week}" / game_id / f"run_id={run_id}"
    return directory / "record.json", directory / "record.sha256"


def write_pregame_run(record: PregameRunRecord) -> Path:
    data_path, hash_path = _paths(record.season, record.week, record.game_id, record.run_id)
    if data_path.is_file():
        raise PregameRunAlreadyExistsError(
            f"Pregame run already exists for game_id={record.game_id!r} run_id={record.run_id!r} - runs are immutable and never overwritten."
        )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(asdict(record), indent=2, sort_keys=True, default=str).encode("utf-8")
    data_path.write_bytes(encoded)
    hash_path.write_text(hashlib.sha256(encoded).hexdigest(), encoding="utf-8")
    return data_path


def list_pregame_runs(season: int, week: int, game_id: str) -> list[str]:
    directory = get_settings().data_dir / PREGAME_RUN_DIRNAME / f"season={season}" / f"week={week}" / game_id
    if not directory.is_dir():
        return []
    return sorted(p.name[len("run_id="):] for p in directory.iterdir() if p.is_dir() and p.name.startswith("run_id="))


def read_pregame_run(season: int, week: int, game_id: str, run_id: str) -> dict:
    data_path, hash_path = _paths(season, week, game_id, run_id)
    content = data_path.read_bytes()
    if hash_path.is_file():
        actual = hashlib.sha256(content).hexdigest()
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if actual != recorded:
            raise ValueError(f"Pregame run {run_id!r} for {game_id!r} does not match its recorded hash - modified after being written.")
    return json.loads(content.decode("utf-8"))


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def select_latest_valid_pregame_snapshot(
    season: int, week: int, game_id: str, now: str,
    max_market_age_hours: float = 24, max_research_age_hours: float = 72, require_research: bool = True,
) -> dict | None:
    candidates = []
    for run_id in list_pregame_runs(season, week, game_id):
        record = read_pregame_run(season, week, game_id, run_id)
        run_dt = _parse(record["run_timestamp"])

        kickoff = record.get("kickoff_timestamp")
        if kickoff is not None and run_dt >= _parse(kickoff):
            continue  # a run at/after kickoff can never be a valid pregame snapshot

        if not record["elo_available"]:
            continue

        if record["market_available"]:
            market_age = compute_age_seconds(record.get("market_snapshot_timestamp"), now)
            if is_stale(market_age, max_market_age_hours):
                continue
        elif require_research is False:
            pass  # market unavailable is tolerated only when the caller doesn't require research either (permissive diagnostic use)

        if require_research:
            if not record["research_available"]:
                continue
            research_age = compute_age_seconds(record.get("research_timestamp"), now)
            if is_stale(research_age, max_research_age_hours):
                continue

        candidates.append((run_dt, record))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]
