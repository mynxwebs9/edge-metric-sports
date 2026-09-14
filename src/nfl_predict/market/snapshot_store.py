"""Phase 5 Steps 1/2/20: the historical `market_snapshot` layer.

Builds a market-data table entirely separate from the Phase 2 football feature tables (per
CLAUDE.md principle 2), from the market fields that have sat unused in the raw `schedules`
snapshots since Phase 1 (`spread_line`, `total_line`, `home_moneyline`, `away_moneyline`,
`home_spread_odds`, `away_spread_odds`, `over_odds`, `under_odds` - see
`src/nfl_predict/features/registry.py`'s `MARKET_FIELD_DENYLIST`, which has always forbidden
these from the independent model but never had anywhere else to live until now).

**What this source actually gives us, verified empirically and against nflverse's own
published field dictionary (`nflreadr::dictionary_schedules`):** exactly ONE row per game,
no sportsbook attribution, and no documented snapshot timing. It is NOT labeled "closing" -
the Phase 5 brief explicitly forbids guessing that, and nflverse's own dictionary does not
say so. Every snapshot from this source is labeled `snapshot_type="UNKNOWN"` and
`sportsbook=None`. Coverage is complete (zero nulls across every market column, every
season 2010-2025) - verified directly against the raw Parquet files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.repositories import ManifestsRepository
from nfl_predict.logging_conf import get_logger
from nfl_predict.market.odds_math import nflverse_spread_to_traditional_home_spread

logger = get_logger(__name__)

SNAPSHOTS_DIRNAME = "market/snapshots/source=nflverse_schedules"
MANIFEST_FILENAME = "provenance.json"

RAW_MARKET_COLUMNS = [
    "game_id", "season", "week", "game_type",
    "spread_line", "home_spread_odds", "away_spread_odds",
    "total_line", "over_odds", "under_odds",
    "home_moneyline", "away_moneyline",
]

MARKET_SNAPSHOT_COLUMNS = [
    "game_id", "season", "week", "season_type",
    "sportsbook", "snapshot_timestamp", "snapshot_type", "market_type",
    "home_spread_traditional", "away_spread_traditional",
    "home_spread_price", "away_spread_price",
    "home_moneyline", "away_moneyline",
    "total_line", "over_price", "under_price",
    "raw_spread_line_nflverse",
    "source", "source_dataset", "source_retrieval_id", "source_retrieved_at", "source_content_sha256",
]

_SEASON_TYPE_MAP = {"REG": "REG", "PRE": "PRE", "WC": "POST", "DIV": "POST", "CON": "POST", "SB": "POST"}


@dataclass(frozen=True)
class MarketIngestResult:
    season: int
    n_games: int
    snapshot_path: Path
    manifest_path: Path
    source_retrieval_id: str


def _season_snapshot_paths(season: int) -> tuple[Path, Path]:
    directory = get_settings().data_dir / SNAPSHOTS_DIRNAME / f"season={season}"
    return directory / "data.parquet", directory / MANIFEST_FILENAME


def build_market_snapshot_for_season(season: int, conn=None) -> pl.DataFrame:
    """Reads the canonical raw `schedules` snapshot for `season` (never re-fetches - Phase 1
    already has it) and derives one `market_snapshot` row per game. Raises if that raw
    snapshot cannot be found - this function never fabricates a value."""
    owns_conn = conn is None
    conn = conn or get_connection()
    init_schema(conn)
    try:
        manifest_row = ManifestsRepository(conn).get_latest_canonical_manifest("nflverse", "schedules", season)
    finally:
        if owns_conn:
            conn.close()

    if manifest_row is None:
        raise ValueError(f"No canonical 'schedules' raw snapshot found for season={season} - ingest it first (Phase 1).")

    raw_path = Path(manifest_row["raw_file_path"])
    raw = pl.read_parquet(raw_path, memory_map=False)
    missing = set(RAW_MARKET_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw schedules snapshot for season={season} is missing expected market column(s): {sorted(missing)}")

    raw = raw.select(RAW_MARKET_COLUMNS)
    home_spread_traditional = [nflverse_spread_to_traditional_home_spread(v) for v in raw["spread_line"].to_list()]

    n = raw.height
    frame = raw.with_columns(
        pl.col("game_type").replace_strict(_SEASON_TYPE_MAP, default=pl.col("game_type")).alias("season_type"),
        pl.Series("home_spread_traditional", home_spread_traditional),
        (-pl.Series("home_spread_traditional", home_spread_traditional)).alias("away_spread_traditional"),
        pl.col("home_spread_odds").alias("home_spread_price"),
        pl.col("away_spread_odds").alias("away_spread_price"),
        pl.col("over_odds").alias("over_price"),
        pl.col("under_odds").alias("under_price"),
        pl.col("spread_line").alias("raw_spread_line_nflverse"),
        pl.lit(None, dtype=pl.Utf8).alias("sportsbook"),
        pl.lit(None, dtype=pl.Utf8).alias("snapshot_timestamp"),  # no per-game timestamp published by this source
        pl.lit("UNKNOWN").alias("snapshot_type"),
        pl.lit("full_game").alias("market_type"),
        pl.lit("nflverse").alias("source"),
        pl.lit("schedules").alias("source_dataset"),
        pl.lit(manifest_row["retrieval_id"]).alias("source_retrieval_id"),
        pl.lit(manifest_row["retrieved_at"]).alias("source_retrieved_at"),
        pl.lit(manifest_row["content_sha256"]).alias("source_content_sha256"),
    )
    return frame.select(MARKET_SNAPSHOT_COLUMNS)


def write_market_snapshot_for_season(season: int) -> MarketIngestResult:
    frame = build_market_snapshot_for_season(season)
    data_path, manifest_path = _season_snapshot_paths(season)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(data_path)

    retrieval_id = frame["source_retrieval_id"][0] if frame.height else None
    manifest = {
        "season": season,
        "n_games": frame.height,
        "source": "nflverse",
        "source_dataset": "schedules",
        "source_retrieval_id": retrieval_id,
        "source_retrieved_at": frame["source_retrieved_at"][0] if frame.height else None,
        "source_content_sha256": frame["source_content_sha256"][0] if frame.height else None,
        "original_fields_used": RAW_MARKET_COLUMNS,
        "snapshot_interpretation": (
            "Exactly one row per game from nflverse's schedules dataset - no sportsbook "
            "attribution, no documented open/mid/pregame/closing timing. Labeled "
            "snapshot_type=UNKNOWN, never CLOSING, per nflreadr::dictionary_schedules (which "
            "documents field meaning, not snapshot timing) and the Phase 5 brief's explicit "
            "instruction not to guess."
        ),
        "game_mapping": "game_id copied verbatim from the raw schedules row - identical to nfl_predict.data.games.normalize_games's join key, no transformation.",
        "price_format": "American odds (integers), e.g. -110, +150.",
        "spread_sign_convention": (
            "raw_spread_line_nflverse: positive means home favored (nflverse convention, "
            "verified against nflreadr::dictionary_schedules). home_spread_traditional: "
            "negative means home favored (bookmaker/docs/MODEL_SPEC.md convention) = "
            "-raw_spread_line_nflverse."
        ),
        "missingness": "0 nulls observed across every market column, every season 2010-2025 (verified directly against the raw Parquet files before this module was written).",
        "sportsbook": "unknown/unattributed - nflverse does not document which book(s) this aggregates or represents.",
        "market_data_denylist_note": "These fields are the same ones nfl_predict.features.registry.MARKET_FIELD_DENYLIST forbids from the independent model - this module is the first and only place they are read for a purpose.",
    }
    import json
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Wrote market_snapshot for season=%d (%d games) to %s", season, frame.height, data_path)
    return MarketIngestResult(season=season, n_games=frame.height, snapshot_path=data_path, manifest_path=manifest_path, source_retrieval_id=retrieval_id)


def read_market_snapshot(seasons: list[int]) -> pl.DataFrame:
    frames = []
    for season in seasons:
        data_path, _ = _season_snapshot_paths(season)
        if data_path.is_file():
            frames.append(pl.read_parquet(data_path, memory_map=False))
    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in MARKET_SNAPSHOT_COLUMNS})
    return pl.concat(frames, how="diagonal_relaxed")
