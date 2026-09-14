"""Phase 6 Step 17: the prospective research ledger.

One immutable, append-only entry per (game_id, run_id), frozen BEFORE kickoff: game,
kickoff, research timestamp, quantitative predictions, market snapshot, research
classification, materiality, sources, unresolved risks. After the game completes, outcomes
are joined SEPARATELY via `join_outcomes`, which reads the frozen ledger and returns a NEW
list of dicts - it never writes an outcome field back into the stored ledger file. This
mirrors Phase 4's prediction-ledger / scoring split and Phase 5's live-snapshot append-only
pattern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nfl_predict.config import get_settings

LEDGER_DIRNAME = "research/prospective_ledger"


class LedgerEntryAlreadyExistsError(Exception):
    """Raised when a (game_id, run_id) entry already exists in the ledger - entries are
    append-only and never overwritten."""


@dataclass(frozen=True)
class ProspectiveLedgerEntry:
    game_id: str
    season: int
    week: int
    research_id: str
    run_id: str
    kickoff_timestamp: str | None
    research_timestamp: str
    elo_predicted_margin: float | None
    elo_home_win_probability: float | None
    ridge_predicted_margin: float | None
    lightgbm_predicted_margin: float | None
    market_home_spread_traditional: float | None
    market_home_moneyline: int | None
    market_snapshot_timestamp: str | None
    research_classification: str
    materiality_level: int
    source_urls: tuple[str, ...]
    unresolved_risks: tuple[str, ...]
    frozen_at: str


def _ledger_paths(season: int, week: int) -> tuple[Path, Path]:
    directory = get_settings().data_dir / LEDGER_DIRNAME / f"season={season}" / f"week={week}"
    return directory / "entries.jsonl", directory / "entries.sha256"


def append_ledger_entry(entry: ProspectiveLedgerEntry) -> None:
    data_path, hash_path = _ledger_paths(entry.season, entry.week)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_path.is_file():
        for line in data_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["game_id"] == entry.game_id and row["run_id"] == entry.run_id:
                raise LedgerEntryAlreadyExistsError(
                    f"Ledger entry already exists for game_id={entry.game_id!r} run_id={entry.run_id!r} "
                    "- prospective ledger entries are append-only and never overwritten."
                )

    new_line = json.dumps(asdict(entry), sort_keys=True, default=str)
    with data_path.open("a", encoding="utf-8") as f:
        f.write(new_line + "\n")

    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    hash_path.write_text(digest, encoding="utf-8")


def read_ledger_entries(season: int, week: int) -> list[dict]:
    data_path, hash_path = _ledger_paths(season, week)
    if not data_path.is_file():
        return []
    if hash_path.is_file():
        actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if actual != recorded:
            raise ValueError(
                f"Prospective ledger for season={season} week={week} does not match its "
                "recorded hash - it was modified after being written."
            )
    return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]


def join_outcomes(entries: list[dict], outcomes_by_game_id: dict[str, dict]) -> list[dict]:
    """Returns a NEW list of dicts with an `outcome` field merged in - never mutates
    `entries` (the frozen ledger rows) or writes anything back to disk. This is the ONLY
    place pregame research and postgame outcomes are ever brought together."""
    return [{**entry, "outcome": outcomes_by_game_id.get(entry["game_id"], {})} for entry in entries]
