"""Phase 5 Steps 1-2/20 orchestrator: builds the historical `market_snapshot` layer for
every season 2010-2025 from the raw `schedules` snapshots Phase 1 already ingested.

    python -m nfl_predict.market.run_market_ingest
"""

from __future__ import annotations

import json

from nfl_predict.logging_conf import get_logger
from nfl_predict.market.snapshot_store import write_market_snapshot_for_season

logger = get_logger(__name__)

SEASONS = list(range(2010, 2026))


def run(seasons: list[int] = SEASONS) -> dict:
    results = []
    for season in seasons:
        result = write_market_snapshot_for_season(season)
        results.append({"season": result.season, "n_games": result.n_games, "source_retrieval_id": result.source_retrieval_id})
    return {"seasons_written": results, "total_games": sum(r["n_games"] for r in results)}


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
